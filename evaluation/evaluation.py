import numpy as np 
import cv2
#

def pnp(p3d,p2d,intrinsics):

    #solve ransac pnp
    retval, rot, trans, inliers = cv2.solvePnPRansac(p3d, p2d, intrinsics, None, flags=cv2.SOLVEPNP_EPNP)

    R = cv2.Rodrigues(rot)[0]  # convert to rotation matrix
    T = trans.reshape(-1, 1)
    rt = np.concatenate((R, T), 1)


    return rt

def project_K(pts_3d,RT,K):
    pts_2d=np.matmul(pts_3d,RT[:,:3].T)+RT[:,3:].T
    pts_2d=np.matmul(pts_2d,K.T)
    pts_2d=pts_2d[:,:2]/pts_2d[:,2:]
    return pts_2d


def projection_2d(pose_pred, pose_targets, model, K, threshold=5):
    # proj_mean_diffs=[]
    # projection_2d_recorder=[]

    model_2d_pred = project_K(model, pose_pred, K)
    model_2d_targets = project_K(model, pose_targets, K)
    proj_mean_diff=np.mean(np.linalg.norm(model_2d_pred - model_2d_targets, axis=-1))

    # proj_mean_diffs.append(proj_mean_diff)
    # projection_2d_recorder.append(proj_mean_diff < threshold)
    return proj_mean_diff < threshold ,proj_mean_diff

def projection_2ds(pose_pred,pose_targets,model,K,threshold=5):
    num_sample=100
    model_2d_pred = project_K(model, pose_pred, K)
    model_2d_targets = project_K(model, pose_targets, K)
    error=[]
    for pred in model_2d_pred[:num_sample]:
        min_error = 10000.0
        for gt in model_2d_targets:
            dist=np.linalg.norm(gt-pred)
            if dist<min_error:
                min_error=dist
        error.append(min_error)

    mean_error=np.mean(error)
    # print(mean_error)
    return (mean_error<=threshold),mean_error





def add_err(gt_pose, est_pose, model):
    def transform_points(points_3d, mat):
        rot = np.matmul(mat[:3, :3], points_3d.transpose())
        return rot.transpose() + mat[:3, 3]
    v_A = transform_points(model, gt_pose)
    v_B = transform_points(model, est_pose)

    v_A = np.array([x for x in v_A])
    v_B = np.array([x for x in v_B])
    return np.mean(np.linalg.norm(v_A - v_B, axis=1))


# def adds_err(gt_pose, est_pose, model, sample_num=100):
#     error = []
#     def transform_points(points_3d, mat):
#         rot = np.matmul(mat[:3, :3], points_3d.transpose())
#         return rot.transpose() + mat[:3, 3]
#     v_A = transform_points(model, gt_pose)
#     v_B = transform_points(model, est_pose)
#     for idx_A, perv_A in enumerate(v_A):
#         if idx_A > sample_num: break
#         min_error_perv_A = 10000.0
#         for idx_B, perv_B in enumerate(v_B):
#             if idx_B > sample_num: break
#             if np.linalg.norm(perv_A - perv_B)<min_error_perv_A:
#                 min_error_perv_A = np.linalg.norm(perv_A - perv_B)
#         error.append(min_error_perv_A)
#     return np.mean(error)

def cm_degree_5_metric(pose_pred, pose_targets):
    """ 5 cm 5 degree metric
    1. pose_pred is considered correct if the translation and rotation errors are below 5 cm and 5 degree respectively
    """
    translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_targets[:, 3]) * 100
    rotation_diff = np.dot(pose_pred[:, :3], pose_targets[:, :3].T)
    trace = np.trace(rotation_diff)
    trace = trace if trace <= 3 else 3
    angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
    return (translation_distance < 5 and angular_distance < 5)


def add_metric(pose_pred, pose_targets, model, diameter, percentage=0.1):
    """ ADD metric
    1. compute the average of the 3d distances between the transformed vertices
    2. pose_pred is considered correct if the distance is less than 10% of the object's diameter
    """
    

    diameter = diameter * percentage
    model_pred = np.dot(model, pose_pred[:, :3].T) + pose_pred[:, 3]
    model_targets = np.dot(model, pose_targets[:, :3].T) + pose_targets[:, 3]
    mean_dist=np.mean(np.linalg.norm(model_pred - model_targets, axis=-1))
    # self.add_recorder.append(mean_dist < diameter)
    # self.add_dists.append(mean_dist)
    return (mean_dist <= diameter), mean_dist

def adds_metric(pose_pred, pose_targets, model, diameter, percentage=0.1):
    num_sample=100
    error=[]
    diameter = diameter * percentage
    model_pred = np.dot(model, pose_pred[:, :3].T) + pose_pred[:, 3]
    model_targets = np.dot(model, pose_targets[:, :3].T) + pose_targets[:, 3]
    for pred in model_pred[:num_sample]:
        min_error=10000.0
        for gt in model_targets:
            dist=np.linalg.norm(gt-pred)
            if dist<min_error:
                min_error=dist
        error.append(min_error)
    mean_error=np.mean(error)
    return (mean_error<=diameter),mean_error


