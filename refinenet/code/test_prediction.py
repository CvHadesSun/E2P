''' to test the keypoints detection prediction result
@ author:cvhadessun
'''

from dataset_utils import *
import numpy as np 
import cv2
import dataset_utils

rot_path='./../data/cat/data/rot35.rot'
tra_path='./../data/cat/data/tra35.tra'
img_path='./../data/cat/data/color35.jpg'
model_path='./../data/cat'
fps_3d_path='./../data/cat'
fps_3d = np.loadtxt(os.path.join(fps_3d_path,'fps_n.txt'))
pose = read_pose(rot_path,tra_path)
kp_2d=project('linemod',pose,fps_3d)
img=cv2.imread(img_path)
model=dataset_utils.load_ply_model(os.path.join(model_path,'mesh.ply'))/1000

# print(fps_3d)

# for kp in fps_3d:
#     for i in range(len(model)):
#         if kp[0] ==model[i][0] and p[1] ==model[i][1] and p[2] ==model[i][2]:
#             print(i)
#             break



Visual2dImg(kp_2d,img)
