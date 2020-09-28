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


# def find_nearest_point_distance(pts1,pts2):
#     '''

#     :param pts1:  pn1,2 or 3
#     :param pts2:  pn2,2 or 3
#     :return:
#     '''
#     idxs=find_nearest_point_idx(pts1,pts2)
#     return np.linalg.norm(pts1[idxs]-pts2,2,1)


def projection_2d(pose_pred, pose_targets, model, K, threshold=5):
    model_2d_pred = project_K(model, pose_pred, K)
    model_2d_targets = project_K(model, pose_targets, K)
    proj_mean_diff=np.mean(np.linalg.norm(model_2d_pred - model_2d_targets, axis=-1))

    # print(proj_mean_diff)

    return proj_mean_diff



