import pickle as pkl
import numpy as np
import warnings
import os

import multiprocessing
from multiprocessing import Process, Manager

from ipc_server import server_process_main
from utils import arg_helper

import src.delta_array_mj as delta_array_mj
warnings.filterwarnings("ignore")

###################################################
# Action / Endpoint Mappings
###################################################
MA_GET_ACTION        = 1
MA_UPDATE_POLICY     = 2
MARB_STORE           = 3
MARB_SAVE            = 4
SAVE_MODEL           = 5
LOAD_MODEL           = 6
LOG_INFERENCE        = 7
TT_GET_ACTION        = 8

OBJ_NAMES = ['star', "block", 'hexagon', 'disc', 'cross', 'diamond', 'triangle', 'parallelogram', 'semicircle', "trapezium"]

###################################################
# Client <-> Server Communication
###################################################
def create_server_process():
    config = arg_helper.create_sac_config()
    parent_conn, child_conn = multiprocessing.Pipe()
    manager = Manager()
    batched_queue = manager.Queue()
    response_dict = manager.dict()
    child_proc = multiprocessing.Process(
        target=server_process_main,
        args=(child_conn, batched_queue, response_dict, config),
        daemon=True
    )
    child_proc.start()
    return parent_conn, batched_queue, response_dict, child_proc, config, manager

def send_request(pipe_conn, action_code, data=None, lock=None, batched_queue=None, response_dict=None):
    with lock:
        request = (action_code, data)
        pipe_conn.send(request)
        response = pipe_conn.recv()
    return response
###################################################
# Test Trajectory
###################################################
def run_test_traj_rb(env_id, sim_len, experiment_data, algo, VideoRecorder, config, inference, pipe_conn, lock):
    print(algo)
    algo_dict = {}
    os.makedirs(f"./data/videos/{config['name']}", exist_ok=True)
    if config['save_vid']:
        recorder = VideoRecorder(output_dir=f"./data/videos/{config['name']}", fps=120)
    else:
        recorder = None
    if recorder is not None:
        n_trials = 1
    else:
        n_trials = 5
        
    print(f"Running Test Trajectories for {algo}")
    for obj_name in OBJ_NAMES:

        print(f"{algo}: Object: {obj_name}")
        algo_dict[obj_name] = {}
        env = delta_array_mj.DeltaArrayRB(config, obj_name)
        with open("./data/test_trajs/test_trajs.pkl", "rb") as f:
            data = pkl.load(f)
                    
        for traj_name, path in data.items():
            
            if recorder:
                recorder.start_recording(filename=f"{algo}_{obj_name}_{traj_name}.mp4")
            algo_dict[obj_name][traj_name] = {}
            
            for run_id in range(n_trials):
                algo_dict[obj_name][traj_name][run_id] = []
                path_cp = path.tolist()
                init_pose = path_cp.pop(0)
                next_pose = path_cp.pop(0)
                env.soft_reset(init=init_pose, goal=next_pose)
                
                try_id = 0
                tries = 1
                
                while len(path_cp) > 0:
                    try_id += 1
                    env.apply_action(env.actions_grasp[:env.n_idxs])
                    env.update_sim(sim_len, recorder)
                    
                    push_states = (algo, env.init_state[:env.n_idxs], env.pos[:env.n_idxs], inference)
                    if algo == "Vis Servo":
                        execute_actions = env.vs_action(random=False)
                        actions = -1*np.ones((env.n_idxs, 3))
                        actions[:, :2] = execute_actions
                    elif algo == "Random":
                        execute_actions = env.vs_action(random=True)
                        actions = -1*np.ones((env.n_idxs, 3))
                        actions[:, :2] = execute_actions
                    else:
                        actions = send_request(pipe_conn, TT_GET_ACTION, push_states, lock)
                        
                        actions = actions[:env.n_idxs]
                        if actions.shape[-1] == 3:
                            active_idxs = np.array(env.active_idxs)
                            inactive_idxs = active_idxs[actions[:, 2] > 0]

                            if len(inactive_idxs) > 0:
                                env.set_z_positions(active_idxs=list(inactive_idxs), low=False)  # Set to high
                            execute_actions = env.clip_actions_to_ws(actions[:, :2])
                        else:
                            # print(actions.shape)
                            execute_actions = env.clip_actions_to_ws(actions)
                            # print(execute_actions.shape)
                            actions = -1*np.ones((env.n_idxs, 3))
                            actions[:, :2] = execute_actions
                            
                        env.final_state[:env.n_idxs, 4:6] = execute_actions
                        
                    env.apply_action(execute_actions)
                    env.update_sim(sim_len, recorder)
                    env.set_rl_states(execute_actions, final=True, test_traj=True)
                    dist, reward = env.compute_reward(actions, inference)
                    print(f"Algo: {algo}, Obj: {obj_name}, Traj: {traj_name}, Run: {run_id}, Attempt: {try_id}, Reward: {reward}")
                    
                    step_data = {
                            'try_id'        : try_id,
                            'init_qpos'     : env.init_qpos,
                            'goal_qpos'     : env.goal_qpos,
                            'final_qpos'    : env.final_qpos,
                            'dist'          : dist,
                            'reward'        : float(reward),
                            'tries'         : tries,
                            'robot_indices' : (env.active_idxs.copy()),
                            'actions'       : actions,
                            'robot_count'   : (env.n_idxs),
                        }
                    
                    algo_dict[obj_name][traj_name][run_id].append(step_data)
                    
                    if (reward > 80) or (tries>3):
                        tries = 1
                        next_pose = path_cp.pop(0)
                        env.soft_reset(goal=next_pose)
                    else:
                        tries+=1
                        env.soft_reset()
                        
                if recorder is not None:
                    recorder.stop_recording()
    
    experiment_data[algo] = algo_dict
        
###################################################
# Main
###################################################
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    from utils.video_utils import VideoRecorder
    
    parent_conn, batched_queue, response_dict, child_proc, config, manager = create_server_process()
    if config['gui'] and config['nenv'] > 1:
        raise ValueError("Cannot run multiple environments with GUI")

    lock = manager.Lock()
        
    current_episode = 0
    n_updates = config['nenv']
    avg_reward = 0
    
    if config['test']:
        send_request(parent_conn, LOAD_MODEL, lock=lock)
        inference = True
    else:
        inference = False

    outer_loops = 0
    if config['test_traj']:
            
        if config['gui']:
            algos = ['MATBC_FT_OG']
        else:
            algos = ["Random", "Vis Servo", "MATSAC", "MATBC", "MATBC_FT", "MATBC_FT_OG", "MATBC_FT_DEC", "MATBC_FT_CEC", "MATBC_FT_MEC"]
        
        experiment_data = manager.dict()
        processes = []
        for i in range(len(algos)):
            p = Process(target=run_test_traj_rb, 
                        args=(i, config['simlen'], experiment_data, algos[i], 
                              VideoRecorder, config, True, parent_conn, lock))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
            
        final_expt_data = dict(experiment_data)
        save_path = "./data/test_trajs/test_traj_data_sel_acts.pkl"
        pkl.dump(final_expt_data, open(save_path, "wb"))
        print(f"Saved Test Trajectory Data at {save_path}")
    else:
        print("Did you want to run main.py?")
        

    send_request(parent_conn, action_code=None, data={"endpoint": "exit"}, lock=lock)
    child_proc.join()
    print("[Main] Done.")
