'''
to process the test label info
'''
import os
import numpy as np
from plyfile import PlyData

#occluded-linemod


def readply(path):
    ply=PlyData.read(path)
    data=ply.elements[0].data
    x=data['x']
    y = data['y']
    z = data['z']
    return np.stack([x,y,z],axis=1)

def occ_get_labelinfo(root_dir,obj_name,img_name):
    #
    modelpath=root_dir+'/models/'+obj_name+'/'+obj_name+'.ply'
    posepath=os.path.join(root_dir,'blender_poses',obj_name)
    num=int(img_name.split('_')[-1].split('.')[0])
    pose_name='pose'+str(num)+'.npy'
    posepath=os.path.join(posepath,pose_name)
    pose=np.load(posepath)
    model=readply(modelpath)
    return model,pose






