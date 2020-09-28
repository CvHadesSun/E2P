import open3d
import trimesh
import numpy as np
import math
import os 

# to proces 3d model dataset

def get_obb(model_path):
    #get the obb of 3d model
    model=trimesh.load(model_path)
    points=model.vertices
    pcd=open3d.geometry.PointCloud()
    pcd.points=open3d.utility.Vector3dVector(points)
    obb=pcd.get_oriented_bounding_box()
    box=np.asarray(obb.get_box_points())
    # print(np.asarray(pcd.points))
    # obb.color = (0,1,0)
    # open3d.visualization.draw_geometries([pcd,obb])
    # print(box)
    diameter=sum(pow(np.max(box,0)-np.min(box,0),2))
    return diameter
# get_obb('./../datasets/ycb_video/data/models/002_master_chef_can/textured.obj')

root_path='./../datasets/ycb_video/data/models'
objs=os.listdir(root_path)
for obj in objs:
    print(obj)
    path=os.path.join(root_path,obj,'textured.obj')
    diameter=get_obb(path)
    save_path=os.path.join(root_path,obj,'diameter.txt')
    with open(save_path,'w') as fp:
        fp.write(str(diameter))
        fp.close


