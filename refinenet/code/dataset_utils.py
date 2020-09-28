'''
@author:sunwanhu
to process the linemod dataset
'''

from plyfile import PlyData
import numpy as np 
import matplotlib.pyplot as plt
import os
import cv2
from pycocotools import mask as maskUtils
from skimage.measure import find_contours
def get_intrinsic_cam(cls_name):
    '''
    get the name of model crosppoding instric of camera
    '''

    intrinsic_matrix = {
        'linemod': np.array([[572.4114, 0., 325.2611],
                              [0., 573.57043, 242.04899],
                              [0., 0., 1.]]),
        'blender': np.array([[700.,    0.,  320.],
                             [0.,  700.,  240.],
                             [0.,    0.,    1.]]),
        'pascal': np.asarray([[-3000.0, 0.0, 0.0],
                              [0.0, 3000.0, 0.0],
                              [0.0,    0.0, 1.0]])
    }

    return intrinsic_matrix[cls_name]

def get_corners_3d(ply_path):
    '''
    get the 3d corners of model from .ply file

    '''

    ply = PlyData.read(ply_path)
    data = ply.elements[0].data 


    x = data['x']
    min_x, max_x = np.min(x), np.max(x)
    y = data['y']
    min_y, max_y = np.min(y), np.max(y)
    z = data['z']
    min_z, max_z = np.min(z), np.max(z)
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])


    return corners_3d

def get_3d_model(ply_path):
    '''
    get the vertex of 3d model
    '''

    ply = PlyData.read(ply_path)
    data = ply.elements[0].data
    x=data['x']
    y=data['y']
    z=data['z']
    
    p_3d=np.array([x,y,z])
    p_3d=p_3d.transpose(1,0)

    
    return p_3d

def load_ply_model(model_path):
    ply = PlyData.read(model_path)
    data = ply.elements[0].data
    x = data['x']
    y = data['y']
    z = data['z']
    return np.stack([x, y, z], axis=-1)

def project(K_type,RT,pts_3d):
    '''
    project the vertex_3d to image
    '''
    intrinsic_matrix=get_intrinsic_cam(K_type)
    pts_2d=np.matmul(pts_3d,RT[:,:3].T)+RT[:,3:].T
    pts_2d=np.matmul(pts_2d,intrinsic_matrix.T)
    pts_2d=pts_2d[:,:2]/pts_2d[:,2:]
    return pts_2d 

def read_pose(rot_path, tra_path):
    rot = np.loadtxt(rot_path, skiprows=1)
    tra = np.loadtxt(tra_path, skiprows=1) / 100.
    return np.concatenate([rot, np.reshape(tra, newshape=[3, 1])], axis=-1)


def Visual2dImg(uv,img):
    #plot scatter figure by uv coordinates
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Scatter Plot')
    plt.xlabel('X')
    plt.ylabel('Y')
    # for p in uv:
    x,y=uv[:,0],uv[:,1]
    ax1.scatter(x,y,c = 'r',marker = '*')
    ax1.imshow(img)
    
    
    #
    # plt.legend('x1')
    #
    plt.show()

def get_translation_transform(blender_model_path,orig_model_path,rotation_transform):
    #get the transform matrix
    blender_model = load_ply_model(blender_model_path)
    orig_model = load_ply_model(orig_model_path)/1000
    # print(blender_model.shape)
    blender_model = np.dot(blender_model,rotation_transform.T)
    translation_transform = np.mean(orig_model,axis=0)-np.mean(blender_model,axis=0)

    return translation_transform


def orig_pose_to_blender_pose(pose,transform_mat):
    #transform the orig pose to blender pose 
    rotation_transform = np.array([[1., 0., 0.],
                                [0., -1., 0.],
                                [0., 0., -1.]])
    rot, tra = pose[:, :3], pose[:, 3]
    tra = tra + np.dot(rot, transform_mat)
    rot = np.dot(rot, rotation_transform)
    return np.concatenate([rot, np.reshape(tra, newshape=[3, 1])], axis=-1)


# def get_mask(projection,img_size):
#     #use the projection to get the full mask
#     mask=np.zeros([img_size[0],img_size[1]],dtype=uint8)
    


def get_object_seg(projection,img_size):
    #get the segmentation of one object
    # print(projection.shape)
    h,w=img_size
    new_mask=np.zeros((h,w),dtype=np.uint8)
    
#    new_mask[projection[:,(1,0)]]=255
    for co in projection:
        if co[1]<h and co[0]<w:
            new_mask[int(co[1]),int(co[0])]=255
    # print(new_mask.shape)
    polygon=find_contours(new_mask,0.5)
    len_list=[len(i) for i in polygon]
    polygon_max=polygon[len_list.index(max(len_list))]
    polygon=np.array(polygon_max)
    _x=polygon[:,0]
    _y=polygon[:,1]
    polygon=np.c_[_y,_x]
    polygon_list=[polygon.flatten().tolist()]
    # cv2.imshow('mask',new_mask)
    # cv2.waitKey(0)
    return polygon_list


def ann2RLE(segm,img_shape):
    h,w=img_shape
    rles=maskUtils.frPyObjects(segm, h, w)
    rle=maskUtils.merge(rles)
    return [rle]

def ann2Mask(rle):
    mask=maskUtils.decode(rle)
    return mask

def area(rle):
    return maskUtils.area(rle)

def ann2box(rle):
    return maskUtils.toBbox(rle)


# # ply_path='D:\\data\\linemod_dataset\\cat\\mesh.ply'
# rot_path='F:\\data\\linemod_dataset\\cat\\data\\rot0.rot'
# tra_path='F:\\data\\linemod_dataset\\cat\\data\\tra0.tra'
# img_path='F:\\data\\linemod_dataset\\cat\\data\\color0.jpg'

# img=plt.imread(img_path)
# root_path="F:\\data\\linemod_dataset\\LINEMOD\\cat"
# blender_model_path=os.path.join(root_path,"cat.ply")
# orig_model_path = os.path.join(root_path,"mesh.ply")
# old_model_path = os.path.join(root_path,"OLDmesh.ply")


# rotation_transform = np.array([[1., 0., 0.],
#                                 [0., -1., 0.],
#                                 [0., 0., -1.]])

# pose=read_pose(rot_path,tra_path)
# print(pose)
# orig_model=load_ply_model(orig_model_path)/1000
# p2d=project('linemod',pose,orig_model)

# Visual2dImg(p2d,img)

# # blender_pose=orig_pose_to_blender_pose(pose)

# transform_mat=get_translation_transform(blender_model_path,orig_model_path,rotation_transform)

# # vertex_3d=get_3d_model(ply_path)
