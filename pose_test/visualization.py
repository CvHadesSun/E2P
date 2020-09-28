import matplotlib.pyplot as plt
from skimage.io import imsave
import matplotlib.patches as patches
import open3d
import os
import trimesh
import numpy as np
import cv2
from utils import occ_get_labelinfo
from ycb_test import get_info
# from evaluation.evaltation import project_K
def project_K(pts_3d,RT,K):
    pts_2d=np.matmul(pts_3d,RT[:,:3].T)+RT[:,3:].T
    pts_2d=np.matmul(pts_2d,K.T)
    pts_2d=pts_2d[:,:2]/pts_2d[:,2:]
    return pts_2d
def get_obb(model):
    #get the obb of 3d model
    # model=trimesh.load(model_path)
    # points=model.vertices
    points=model
    pcd=open3d.geometry.PointCloud()
    pcd.points=open3d.utility.Vector3dVector(points)
    obb=pcd.get_axis_aligned_bounding_box()
    # obb=pcd.get_oriented_bounding_box()
    box=np.asarray(obb.get_box_points())
    
    return box

def draw_bbox(img,bb8_pred,bb8_gt,save_path):
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _, ax = plt.subplots(1)
    # print(bb8_pred.shape)
    ax.imshow(img)

    ax.add_patch(
        patches.Polygon(xy=bb8_pred[0,0][[0,1,7,2,0,3,6,1]], fill=False, linewidth=1, edgecolor='b'))
    ax.add_patch(
        patches.Polygon(xy=bb8_pred[0,0][[5,4,6, 3, 5, 2, 7, 4]], fill=False, linewidth=1, edgecolor='b'))
    # if corners_targets is not None:
    ax.add_patch(
        patches.Polygon(xy=bb8_gt[0,0][[0,1,7,2,0,3,6,1]], fill=False, linewidth=1, edgecolor='r'))
    ax.add_patch(
        patches.Polygon(xy=bb8_gt[0,0][[5,4,6, 3, 5, 2, 7, 4]], fill=False, linewidth=1, edgecolor='r'))
    # ax.plot(x=bb8_pred[0,0,:,0],y=bb8[0,0,:,1])
    plt.savefig(save_path)
    plt.close()
#bb8_pred


def draw_multi_bbox(img,bb8_pred,bb8_gt,save_path):
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _, ax = plt.subplots(1)
    # print(bb8_pred.shape)
    ax.imshow(img)
    num=bb8_pred.shape[0]
    for i in range(num):

        ax.add_patch(
            patches.Polygon(xy=bb8_pred[i,0][[0,1,7,2,0,3,6,1]], fill=False, linewidth=1, edgecolor='b'))
        ax.add_patch(
            patches.Polygon(xy=bb8_pred[i,0][[5,4,6, 3, 5, 2, 7, 4]], fill=False, linewidth=1, edgecolor='b'))
        # if corners_targets is not None:
        ax.add_patch(
            patches.Polygon(xy=bb8_gt[i,0][[0,1,7,2,0,3,6,1]], fill=False, linewidth=1, edgecolor='r'))
        ax.add_patch(
            patches.Polygon(xy=bb8_gt[i,0][[5,4,6, 3, 5, 2, 7, 4]], fill=False, linewidth=1, edgecolor='r'))
    # ax.plot(x=bb8_pred[0,0,:,0],y=bb8[0,0,:,1])
    plt.savefig(save_path)
    plt.close()
# root_path='./../datasets/ycb_video/data/models'
# objs=os.listdir(root_path)
# for obj in objs:
#     print(obj)
#     path=os.path.join(root_path,obj,'textured.obj')
#     box=get_obb(path)
#     img=cv2.imread('./../datasets/ycb_video/data/train_data/0000049-color.png')
#     draw_bbox(img,box[None,None,...],box)
#     # save_path=os.path.join(root_path,obj,'diameter.txt')
#     # with open(save_path,'w') as fp:
#     #     fp.write(str(diameter))
#     #     fp.close
#     print(box)
#     break

# root_dir='./../datasets/ycb_video/data'
# obj_name='025_mug'
# img_name='0000001-color.png'

# img=cv2.imread(os.path.join(root_dir,'test_data',img_name))
# pose_gt, model,_,K,_=get_info(root_dir,obj_name,14,img_name)
# box=get_obb(model)
# box_2d=project_K(box,pose_gt,K)
# draw_bbox(img,box_2d[None,None,...],box_2d[None,None,...])
# def vis(root_dir,obj_id,img_name):
#     obj = {1: "002_master_chef_can",
#            2: "003_cracker_box",
#            3: "004_sugar_box",
#            4: "005_tomato_soup_can",
#            5: "006_mustard_bottle",
#            6: "007_tuna_fish_can",
#            7: "008_pudding_box",
#            8:"009_gelatin_box",
#            9:"010_potted_meat_can",
#            10:"011_banana",
#            11:"019_pitcher_base",
#            12:"021_bleach_cleanser",
#            13:"024_bowl",
#            14:"025_mug",
#            15:"035_power_drill",
#            16:"036_wood_block",
#            17:"037_scissors",
#            18:"040_large_marker",
#            19:"051_large_clamp",
#            20:"052_extra_large_clamp",
#            21:"061_foam_brick"}