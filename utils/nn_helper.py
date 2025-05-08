import numpy as np
from scipy.spatial import ConvexHull, KDTree
from shapely.geometry import Polygon, Point
import pandas as pd
from copy import deepcopy

class NNHelper:
    def __init__(self, plane_size, real_or_sim="real"):
        self.rb_pos_pix = np.zeros((8,8,2))
        self.rb_pos_world = np.zeros((8,8,2))
        self.kdtree_positions_pix = np.zeros((64, 2))
        self.kdtree_positions_world = np.zeros((64, 2))
        for i in range(8):
            for j in range(8):
                if real_or_sim=="real":
                    if i%2!=0:
                        finger_pos = np.array((i*0.0375, j*0.043301 - 0.02165))
                        self.rb_pos_world[i,j] = np.array((i*0.0375, j*0.043301 - 0.02165))
                    else:
                        finger_pos = np.array((i*0.0375, j*0.043301))
                        self.rb_pos_world[i,j] = np.array((i*0.0375, j*0.043301))
                else:
                    if i%2!=0:
                        finger_pos = np.array((i*0.0375, j*0.043301 - 0.02165))
                        self.rb_pos_world[i,j] = np.array((i*0.0375, j*0.043301 - 0.02165))
                    else:
                        finger_pos = np.array((i*0.0375, j*0.043301))
                        self.rb_pos_world[i,j] = np.array((i*0.0375, j*0.043301))
                self.kdtree_positions_world[i*8 + j, :] = self.rb_pos_world[i,j]
        
                finger_pos[0] = (finger_pos[0] - plane_size[0][0])/(plane_size[1][0]-plane_size[0][0])*1080 - 0
                if real_or_sim=="real":
                    finger_pos[1] = 1920 - (finger_pos[1] - plane_size[0][1])/(plane_size[1][1]-plane_size[0][1])*1920
                else:
                    finger_pos[1] = 1920 - (finger_pos[1] - plane_size[0][1])/(plane_size[1][1]-plane_size[0][1])*1920
                
                self.rb_pos_pix[i,j] = finger_pos
                self.kdtree_positions_pix[i*8 + j, :] = self.rb_pos_pix[i,j]
        
        self.cluster_centers = None

    def find_robots_outside_non_convex(self, sampled_boundary_points_world, finger_radius = 0.03):
        sampled_kdtree = KDTree(sampled_boundary_points_world)
        distances, nearest_boundary_indices_all = sampled_kdtree.query(
            self.kdtree_positions_world, k=1, workers=-1
        )
        proximity_mask = distances <= finger_radius
        proximity_candidates_indices = np.where(proximity_mask)[0]

        if proximity_candidates_indices.shape[0] == 0:
            return np.array([], dtype=int), np.empty((0, 2)), np.array([], dtype=int)

        polygon = Polygon(sampled_boundary_points_world)
        candidate_robot_points = [Point(p) for p in self.kdtree_positions_world[proximity_candidates_indices]]
        is_inside_mask = np.array([polygon.contains(p) for p in candidate_robot_points], dtype=bool)

        final_active_robot_indices_provisional = proximity_candidates_indices[~is_inside_mask]
        if final_active_robot_indices_provisional.shape[0] == 0:
            return np.array([], dtype=int), np.empty((0, 2)), np.array([], dtype=int)

        candidate_distances = distances[final_active_robot_indices_provisional]
        candidate_boundary_indices = nearest_boundary_indices_all[final_active_robot_indices_provisional]

        df = pd.DataFrame({
            'robot_idx': final_active_robot_indices_provisional,
            'boundary_idx': candidate_boundary_indices,
            'distance': candidate_distances
        })
        idx_min_dist = df.loc[df.groupby('boundary_idx')['distance'].idxmin()]
        active_robot_indices_final = idx_min_dist['robot_idx'].to_numpy(dtype=int)
        matched_boundary_indices_final = idx_min_dist['boundary_idx'].to_numpy(dtype=int)

        if active_robot_indices_final.shape[0] == 0:
            return np.array([], dtype=int), np.empty((0, 2)), np.array([], dtype=int)

        matched_boundary_pts_final = sampled_boundary_points_world[matched_boundary_indices_final]
        return active_robot_indices_final, matched_boundary_pts_final, matched_boundary_indices_final

    def get_nn_robots_objs(self, boundary_pts, world=True):
        hull = ConvexHull(boundary_pts)
        hull = self.expand_hull(hull, world=world)  # custom user function
        A, b = hull.equations[:, :-1], hull.equations[:, -1:]
        
        kdtree_poses = deepcopy(self.kdtree_positions_world) if world else deepcopy(self.kdtree_positions_pix)
        main_kdtree = KDTree(kdtree_poses)

        eps = np.finfo(np.float32).eps
        dub = 0.04 if world else 30

        distances, idx_candidates = main_kdtree.query(boundary_pts, k=8, distance_upper_bound=dub, workers=1)
        valid_indices = idx_candidates[~np.isinf(distances)]
        unique_indices = np.unique(valid_indices)

        pos_world = deepcopy(self.rb_pos_world[unique_indices // 8, unique_indices % 8])
        inside_mask = np.all(pos_world @ A.T + b.T < eps, axis=1)

        boundary_kdtree = KDTree(boundary_pts)
        nearest_neighbors = {}
        for robot_idx, is_inside in zip(unique_indices, inside_mask):
            robot_pos = kdtree_poses[robot_idx]

            if not is_inside:
                _, nearest_bd_idx = boundary_kdtree.query(robot_pos[None, :], k=1)
                nearest_neighbors[robot_idx] = nearest_bd_idx[0]

        final_robot_indices = list(nearest_neighbors.keys())            # len = K
        final_boundary_indices = list(nearest_neighbors.values())       # len = K
        final_bd_pts = boundary_pts[final_boundary_indices]            # shape: (K, 2)
        return final_robot_indices, final_bd_pts, final_boundary_indices
    
    def expand_hull(self, hull, world=True):
        """
        Expands the convex hull by the radius of the robot
        """
        if world:
            robot_radius = 0.005
        else:
            robot_radius = 30
        expanded_hull_vertices = []
        for simplex in hull.simplices:
            v1, v2 = hull.points[simplex]
            
            edge_vector = v2 - v1
            normal_vector = np.array([-edge_vector[1], edge_vector[0]])
            normal_vector /= np.linalg.norm(normal_vector)
            
            expanded_v1 = v1 + robot_radius * normal_vector
            expanded_v2 = v2 + robot_radius * normal_vector
            expanded_hull_vertices.extend([expanded_v1, expanded_v2])

        return ConvexHull(expanded_hull_vertices)
        