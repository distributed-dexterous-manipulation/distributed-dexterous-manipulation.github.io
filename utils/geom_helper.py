import numpy as np
from scipy.interpolate import interp1d

lower_green_filter = np.array([35, 50, 50])
upper_green_filter = np.array([85, 255, 255])

def get_2Dtf_matrix(x, y, yaw):
    return np.array([[np.cos(yaw), -np.sin(yaw), x],
                     [np.sin(yaw), np.cos(yaw), y],
                     [0, 0, 1]])
    
def transform2D_pts(bd_pts, M):
    homo_pts = np.hstack([bd_pts, np.ones((bd_pts.shape[0], 1))])
    tfed_bd_pts = np.dot(M, homo_pts.T).T
    return tfed_bd_pts[:, :2]

def get_tfed_2Dpts(init_bd_pts, init_pose, goal_pose):
    M_init = get_2Dtf_matrix(*init_pose)
    M_goal = get_2Dtf_matrix(*goal_pose)
    M = M_goal @ np.linalg.inv(M_init)
    return transform2D_pts(init_bd_pts, M)

def transform_pts_wrt_com(points, transform, com):
    points_h = np.hstack([points - com, np.zeros((points.shape[0], 1)), np.ones((points.shape[0], 1))])
    transformed_points_h = points_h @ transform.T
    transformed_points = transformed_points_h[:, :2] + com
    return transformed_points

# Method 2: Angle-Based Ordering
def sample_boundary_points(boundary_points: np.ndarray, n_samples: int) -> np.ndarray:
    centroid = np.mean(boundary_points, axis=0)
    angles = np.arctan2(boundary_points[:,1] - centroid[1], boundary_points[:,0] - centroid[0])
    sorted_indices = np.argsort(angles)
    ordered_points = boundary_points[sorted_indices]
    ordered_points = np.vstack([ordered_points, ordered_points[0]])
    diffs = np.diff(ordered_points, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    cumulative_dist = np.concatenate(([0], np.cumsum(segment_lengths)))
    cumulative_dist /= cumulative_dist[-1]
    fx = interp1d(cumulative_dist, ordered_points[:, 0], kind='linear')
    fy = interp1d(cumulative_dist, ordered_points[:, 1], kind='linear')
    t = np.linspace(0, 1, n_samples)
    return np.column_stack([fx(t), fy(t)])

def icp_rot_with_correspondence(src, tgt):
    # Find nearest neighbors in tgt for each point in src
    tgt_matched = np.zeros_like(src)
    for i in range(len(src)):
        distances = np.linalg.norm(tgt - src[i], axis=1)
        tgt_matched[i] = tgt[np.argmin(distances)]
    
    H = src.T @ tgt_matched
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    return R

def transform_boundary_points(init_bd_pts, goal_bd_pts, init_nn_bd_pts):
    com0 = np.mean(init_bd_pts, axis=0)
    com1 = np.mean(goal_bd_pts, axis=0)
    
    src = init_bd_pts - com0
    tgt = goal_bd_pts - com1
    
    R = icp_rot_with_correspondence(src, tgt)
    
    transformed_nn = (R @ (init_nn_bd_pts - com0).T).T + com1
    return transformed_nn

def random_resample_boundary_points(init_bd_pts: np.ndarray, goal_bd_pts: np.ndarray):
    n1, n2 = len(init_bd_pts), len(goal_bd_pts)
    target_size = min(n1, n2)
    
    if n1 > target_size:
        indices = np.random.choice(n1, size=target_size, replace=False)
        init_bd_pts = init_bd_pts[indices]
    
    if n2 > target_size:
        indices = np.random.choice(n2, size=target_size, replace=False)
        goal_bd_pts = goal_bd_pts[indices]
        
    return init_bd_pts, goal_bd_pts

        