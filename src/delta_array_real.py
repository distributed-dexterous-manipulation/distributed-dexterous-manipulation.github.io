import numpy as np
import time
import socket
import cv2
import threading
import math

from delta_array_utils.Prismatic_Delta import Prismatic_Delta
from delta_array_utils.get_coords import RoboCoords
from delta_array_utils.DeltaRobotAgent import DeltaArrayAgent
import delta_array_utils.serial_robot_mapping as srm

import utils.nn_helper as nn_helper
from utils.vision_utils import VisUtils
import utils.geom_helper as geom_helper
from utils.video_utils import VideoRecorder

np.set_printoptions(precision=4)

LOW_Z = 9.8
MID_Z = 7.5
HIGH_Z = 5.5
BUFFER_SIZE = 20
theta = -np.pi / 2
R_z = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta),  np.cos(theta), 0],
    [0,             0,              1]
])

current_frame = None
global_bd_pts, global_yaw, global_com = None, None, None
# lock = threading.Lock()

def capture_and_convert(stop_event, current_traj, lock, rl_device, trad, plane_size, save_vid, vid_name=None, n_obj=1):
    global current_frame, global_bd_pts, global_yaw, global_com
    
    vis_utils = VisUtils(device=rl_device, traditional=trad, plane_size=plane_size)
    video_recorder = None
    if save_vid:
        video_recorder = VideoRecorder(output_dir="./data/videos/real", fps=120, resolution=(1920, 1080))
        video_recorder.start_recording(vid_name)

    camera_matrix = np.load("./utils/calibration_data/camera_matrix.npy") 
    dist_coeffs = np.load("./utils/calibration_data/dist_coeffs.npy")

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    marker_size = 0.015

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            continue

        bd_pts_world = vis_utils.get_bd_pts(frame, total_pts=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None and len(corners) > 0:
            rvecs, _, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)

            rvec = rvecs[0, 0, :]
            R, _ = cv2.Rodrigues(rvec)
            R =  R @ R_z
            yaw = math.atan2(R[1, 0], R[0, 0])
            
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            com_world = np.mean(bd_pts_world, axis=0)
            with lock:
                global_com = com_world
                global_yaw = yaw
                global_bd_pts = bd_pts_world

                current_frame = frame.copy()
                
            if current_frame is not None:
                cv2.imshow('Stream', current_frame)

            if video_recorder is not None:
                video_recorder.add_frame(cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB))
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if video_recorder is not None:
        video_recorder.stop_recording()
    print("[Child] Exiting child process safely.")
    
def start_capture_thread(current_traj, lock, rl_device, trad, plane_size, save_vid, vid_name=None, n_obj=1):
    stop_event = threading.Event()

    capture_thread = threading.Thread(
        target=capture_and_convert,
        args=(stop_event, current_traj, lock, rl_device, trad, plane_size, save_vid, vid_name, n_obj),
        daemon=True  # so it won’t block your main thread from exiting if you forget to join
    )
    capture_thread.start()
    return stop_event, capture_thread
    
def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def delta_angle(yaw1, yaw2):
    return abs(np.arctan2(np.sin(yaw1 - yaw2), np.cos(yaw1 - yaw2)))
    
class DeltaArrayReal:
    def __init__(self, config):
        self.max_agents = 64
        self.img_size = np.array((1080, 1920))
        self.plane_size = np.array([(0.009,  -0.034),(0.24200, 0.376)])
        self.delta_plane_x = self.plane_size[1][0] - self.plane_size[0][0]
        self.delta_plane_y = self.plane_size[1][1] - self.plane_size[0][1]
        self.delta_plane = np.array((self.delta_plane_x, -self.delta_plane_y))
        self.nn_helper = nn_helper.NNHelper(self.plane_size, real_or_sim="real")

        """ Real World Util Vars """
        self.NUM_MOTORS = 12
        self.to_be_moved = []
        s_p = 1.5 #side length of the platform
        s_b = 4.3 #side length of the base
        length = 4.5 #length of leg attached to platform
        self.Delta = Prismatic_Delta(s_p, s_b, length)
        self.RC = RoboCoords()
        self.active_idxs = []
        self.active_IDs = set()
        self.n_idxs = 0
        self.all_robots = np.arange(64)
        self.prev_robot_pos = np.zeros((64, 2))
        
        self.traditional = config['traditional']
        self.obj_name = config['obj_name']
        
        """ Setup Delta Robot Agents """
        self.delta_agents = {}
        self.setup_delta_agents()
        self.config = config

    def setup_delta_agents(self):
        self.delta_agents = {}
        for i in range(1, 17):
            try:
                ip_addr = srm.inv_delta_comm_dict[i]
                esp01 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                esp01.connect((ip_addr, 80))
                esp01.settimeout(0.05)
                self.delta_agents[i-1] = DeltaArrayAgent(esp01, i)
            except Exception as e:
                print("Error at robot ID: ", i)
                raise e
        self.reset()
        
    def reconnect_delta_agents(self):
        for i in range(1, 17):
            self.delta_agents[i-1].esp01.close()
            del self.delta_agents[i-1]
            
        self.delta_agents = {}
        for i in range(1, 17):
            try:
                ip_addr = srm.inv_delta_comm_dict[i]
                esp01 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                esp01.connect((ip_addr, 80))
                esp01.settimeout(0.05)
                self.delta_agents[i-1] = DeltaArrayAgent(esp01, i)
            except Exception as e:
                print("Error at robot ID: ", i)
                raise e
        self.reset()
    
    def pract_grasp(self, actions, y_mult, idx):
        mid_pos = np.array([-0.6 * actions[0], -0.6*y_mult*actions[1], LOW_Z])
        mid_pos2 = np.array([0, 0, LOW_Z])
        end_pos = np.array([0.95*actions[0], 0.95*y_mult * actions[1], LOW_Z])
        traj = np.zeros((20, 3))
        
        traj[0] = mid_pos
        traj[1] = mid_pos2
        traj[2:] = np.linspace(mid_pos2, end_pos, num=18)
        self.prev_robot_pos[idx] = end_pos[:2]
        return traj.tolist()
        
    def pract_push(self, actions, y_mult, idx):
        start_pos = np.array([*self.prev_robot_pos[idx][:2], LOW_Z])
        push_pos = np.array([0.99*actions[0], 0.99*y_mult*actions[1], LOW_Z])
        mid_pos = np.array([0.99*actions[0], 0.99*y_mult*actions[1], MID_Z])
        mid_pos_2 = np.array([0, 0, MID_Z])
        end_pos = np.array([0, 0, HIGH_Z])
        traj = np.zeros((20, 3))
        
        # Robot goes slightly away along normal from boundary point at zlow
        traj[:17] = np.linspace(start_pos, push_pos, num=17)
        
        # Robot interpolates a trajectory to the actual action i.e. near the bd pt
        traj[17] = mid_pos
        traj[18] = mid_pos_2
        traj[19:] = end_pos
        return traj.tolist()
        
    def move_robots(self, active_idxs, actions, z_level, push=False, practicalize=False):
        actions = self.clip_actions_to_ws(95*actions.copy())
        for i, idx in enumerate(active_idxs):
            y_mult = -1 # cos the y-axis is flipped for the upside-down array configuration
            
            if practicalize:
                if push:
                    traj = self.pract_push(actions[i], y_mult, idx)
                else:
                    traj = self.pract_grasp(actions[i], y_mult, idx)
            else:
                traj = [[actions[i][0], y_mult*actions[i][1], z_level] for _ in range(20)]
                
            idx2 = (idx//8, idx%8)
            self.delta_agents[self.RC.robo_dict_inv[idx2] - 1].save_joint_positions(idx2, traj)
            self.active_IDs.add(self.RC.robo_dict_inv[idx2])

        for i in self.active_IDs:
            self.delta_agents[i - 1].move_useful()
            self.to_be_moved.append(self.delta_agents[i - 1])

        self.wait_until_done()

    def reset(self):
        self.raw_rb_pos = None
        self.init_state = np.zeros((64, 6))
        self.final_state = np.zeros((64, 6))
        self.actions = np.zeros((64, 2))
        self.pos = np.zeros(64)
        self.active_idxs = []
        self.n_idxs = 0

        for i in set(self.RC.robo_dict_inv.values()):
            self.delta_agents[i-1].reset()
            self.to_be_moved.append(self.delta_agents[i-1])

        self.wait_until_done()
    
    def wait_until_done(self):
        done_moving = False
        start_time = time.time()
        while not done_moving:
            for i in self.to_be_moved:
                try:
                    received = i.esp01.recv(BUFFER_SIZE)
                    ret = received.decode().strip()
                    if ret == "A":
                        i.done_moving = True
                except Exception as e:
                    time.sleep(0.1)
                    pass
                
            bool_dones = [i.done_moving for i in self.to_be_moved]
            done_moving = all(bool_dones)
            # Break if no communication happens in 15 seconds
            if time.time() - start_time > 15:
                print("Timeout exceeded while waiting for agents to complete.")
                done_moving = True
        time.sleep(0.1)
        for i in self.delta_agents:
            self.delta_agents[i].done_moving = False
        del self.to_be_moved[:]
        self.active_IDs.clear()
    
    def set_z_positions(self, active_idxs=None, low=True):
        if active_idxs is None:
            active_idxs = self.all_robots.copy()
        actions = []
        for idx in active_idxs:
            actions.append([0, 0])
        self.move_robots(active_idxs, np.array(actions), LOW_Z if low else HIGH_Z, practicalize=True)
        
    def clip_actions_to_ws(self, actions):
        return self.Delta.clip_points_to_workspace(actions)
    
    def vs_action(self, act_grasp, random=False):
        if random:
            actions = np.random.uniform(-0.03, 0.03, size=(self.n_idxs, 2))
        else:
            self.sf_bd_pts, self.sf_nn_bd_pts = self.get_current_bd_pts()
            displacement_vectors = self.goal_nn_bd_pts - self.sf_nn_bd_pts
            actions = act_grasp + displacement_vectors
        return actions
    
    def set_rl_states(self, actions=None, final=False, test_traj=False):
        if final:
            bdpts, yaw, xy = self.get_bdpts_and_pose()
            self.final_qpos = [*xy, yaw]
            self.final_bd_pts, self.final_nn_bd_pts = self.get_current_bd_pts()
            self.final_bd_pts, self.final_nn_bd_pts = self.get_current_bd_pts()
            self.final_state[:self.n_idxs, :2] = self.final_nn_bd_pts - self.raw_rb_pos
            self.final_state[:self.n_idxs, 4:6] = actions[:self.n_idxs]
        else:
            
            self.n_idxs = len(self.active_idxs)
            self.pos[:self.n_idxs] = self.active_idxs.copy()
            self.raw_rb_pos = self.nn_helper.kdtree_positions_world[self.active_idxs]
            
            self.init_state[:self.n_idxs, :2] = self.init_nn_bd_pts - self.raw_rb_pos
            self.init_state[:self.n_idxs, 2:4] = self.goal_nn_bd_pts - self.raw_rb_pos
            
            acts = self.clip_actions_to_ws(self.init_nn_bd_pts - self.raw_rb_pos)
            self.init_state[:self.n_idxs, 4:6] = acts
            self.final_state[:self.n_idxs, 2:4] = self.init_state[:self.n_idxs, 2:4].copy()
            
            return acts.copy()
        
    def get_current_bd_pts(self):
        bd_pts = None
        while bd_pts is None:
            bd_pts = global_bd_pts.copy()
        nn_bd_pts = geom_helper.transform_boundary_points(self.init_bd_pts.copy(), bd_pts, self.init_nn_bd_pts.copy())
        return bd_pts, nn_bd_pts
        
    def get_bdpts_and_pose(self):
        bd_pts, yaw, com = global_bd_pts.copy(), global_yaw, global_com.copy()
        return bd_pts, yaw, com
        
    def set_goal_nn_bd_pts(self):
        self.goal_nn_bd_pts = geom_helper.transform_boundary_points(self.init_bd_pts.copy(), self.goal_bd_pts.copy(), self.init_nn_bd_pts.copy())
    
    def get_active_idxs(self):
        idxs, self.init_nn_bd_pts, _ = self.nn_helper.find_robots_outside_non_convex(self.init_bd_pts)
        self.active_idxs = list(idxs)
        
    def compute_reward(self, actions):
        dist = np.mean(np.linalg.norm(self.goal_nn_bd_pts - self.final_nn_bd_pts, axis=1))
        ep_reward = 1 / (10000 * dist**3 + 0.01)
        return dist, ep_reward
    
    def soft_reset(self, init_2Dpose=None, goal_2Dpose=None):
        if goal_2Dpose is None:
            pass
        else:
            self.reconnect_delta_agents()
            
        self.raw_rb_pos = None
        self.init_state = np.zeros((64, 6))
        self.final_state = np.zeros((64, 6))
        self.actions = np.zeros((64, 2))
        self.pos = np.zeros(64)
        self.active_idxs = []
        self.n_idxs = 0
        
        if init_2Dpose is not None:
            self.init_2Dpose = init_2Dpose
            delta_x = 99
            delta_y = 99
            delta_yaw = 99
            while (delta_x > 0.006) or (delta_y > 0.006) or (delta_yaw > 0.1):
                self.init_bd_pts, yaw, com = self.get_bdpts_and_pose()
                delta_x = abs(com[0] - init_2Dpose[0])
                delta_y = abs(com[1] - init_2Dpose[1])
                delta_yaw = delta_angle(yaw, init_2Dpose[2])
                time.sleep(0.1)
                print(f"x_err: {delta_x}; y_err: {delta_y}; yaw_err: {delta_yaw}")
                
        self.init_bd_pts, yaw, com = self.get_bdpts_and_pose()
        self.init_qpos = [*com, yaw]
        self.get_active_idxs()
        
        if goal_2Dpose is not None:
            self.goal_bd_pts = geom_helper.get_tfed_2Dpts(self.init_bd_pts, self.init_qpos, goal_2Dpose)
        
        self.set_goal_nn_bd_pts()
        act_grasp = self.set_rl_states()
        return act_grasp
        
    def rollout(self, act_grasp, actions):
        if actions.shape[-1] == 3:
            self.final_state[:self.n_idxs, 4:6] = actions[:, :2]
            active_idxs = np.array(self.active_idxs)
            sel_idxs = actions[:, 2] < 0
            active_idxs = active_idxs[sel_idxs]
            execute_actions = actions[sel_idxs, :2]
            self.move_robots(active_idxs, act_grasp[sel_idxs], LOW_Z, practicalize=True)
        else:
            active_idxs = np.array(self.active_idxs)
            sel_idxs = np.ones_like(actions[:, 0], dtype=bool)
            execute_actions = actions.copy()
            self.move_robots(active_idxs, act_grasp, LOW_Z, practicalize=True)
                
        self.move_robots(active_idxs, execute_actions, LOW_Z, push=True, practicalize=True)
        self.set_rl_states(actions[:, :2], final=True, test_traj=True)
        dist, reward = self.compute_reward(actions)
        return dist, reward