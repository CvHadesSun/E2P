'''
the tools to process the linemod dataset or occluded-linemod dataset
'''

from plyfile import PlyData
import numpy as np 
import matplotlib.pyplot as plt
import os
import cv2
import glob
# from dataset_utils import load_ply_model
# from dataset_utils import read_pose
# from dataset_utils import get_object_seg
from .dataset_utils import *


class Linemod(object):
    '''
    input the linemod data root path and object name 
    get the all train RGB images' name of the object 
    '''
    def __init__(self,root_path,object_name):
        self.path=os.path.join(root_path,object_name)
        # self.img_names=self.get_all_img(self.path)
        self.categories_info={"num_object":1,"name":[object_name]}
        
    def get_all_img(self,path):
        data_path=os.path.join(path,'data')
        img_list= glob.glob(os.path.join(data_path,'color*'))
        img_list=[c.split('/')[-1] for c in img_list] #for linux os 
        # print(img_list)
        #return the images' name
        return img_list
    # def get_img(self,filename):
    #     #get image path from the txt file 
    #     filepath=os.path.join(self.path,filename)
    #     with open(filepath,'r') as fp:


class LabelInfo(object):
    '''
    process one image to get the annotation info
    :model
    :area
    :pose
    :segmentation
    :bbox
    :keypoints
    ...
    '''
    
    def __init__(self,path,img_name):
        #the path is the object root path
        #
        self.path=path
        # self.cls_name=cls_name
        self.num_object=1
        self.model=self.get_model(path)
        self.category_ids=[1]
        self.num_objects=1
        self.pose=0
        self.fps={}
        self.label=self.get_label(img_name)


    def get_model(self,path):
        #load the ply file of model
        model_path=os.path.join(path,'mesh.ply')
        model_3d=load_ply_model(model_path)/1000
        return model_3d

    def get_label(self,img_name):
        #return the label info of one image 
        label_info={}
        data_path=os.path.join(self.path,'data')

        # print(data_path)
        # print(data_path)
        # print(img_name)
        img=cv2.imread(os.path.join(data_path,img_name))
        h,w,_=img.shape
        rot_name=img_name.replace('color','rot').replace('.jpg','.rot')
        tra_name=img_name.replace('color','tra').replace('.jpg','.tra')
        pose=read_pose(os.path.join(data_path,rot_name),os.path.join(data_path,tra_name))

        self.pose=pose
        model=self.model
        model_pro=project('linemod',pose,model)
        #get segmentation
        segm=get_object_seg(model_pro,[h,w])
        #use segm to get the area and bbox
        rle=ann2RLE(segm,[h,w])
        _area=area(rle)
        bbox=ann2box(rle)
        #2d keypoints
        corner_3d=np.loadtxt(os.path.join(self.path,'corners.txt'))/1000
        fps_3d=np.loadtxt(os.path.join(self.path,'fps_8.txt')) # already model devided by 1000
        self.fps['fps_8']=fps_3d
        # can give corner_3d or fps_n to kp_3d to become keypoints to train 
        kp_3d=fps_3d
        kp_2d=project('linemod',pose,kp_3d)
        #extern one dimension for kp_2d about the keypoint is visable or not
        num_keypoints=kp_2d.shape[0]
        extern_d=np.array([2]*num_keypoints).reshape(num_keypoints,1)
        kp_2d_val=np.c_[kp_2d,extern_d]
        #write the label info to label_info
        label_info['model']=model
        label_info['pose']=pose
        label_info['keypoints']=kp_2d_val.flatten().tolist()
        label_info['segmentation']=segm
        label_info['area']=float(_area[0])
        label_info['bbox']=bbox[0].tolist()
        # print(type(segm))
        # print(type(float(_area[0])))
        # print(type(bbox))
        #
        label={}
        label['1']=label_info
        return label

        


class LabelInfoGenerateData(object):
    '''
    process one image to get the annotation info
    :model
    :area
    :pose
    :segmentation
    :bbox
    :keypoints
    ...
    '''
    def __init__(self,path,img_name):
        #the path is the object root path
        #
        self.path=path  #root folder
        # self.cls_name=cls_name
        self.num_object=1
        self.model=self.get_model(path)
        self.category_ids=[1]
        self.num_objects=1
        self.label=self.get_label(img_name)


    def get_model(self,path):
        #load the ply file of model
        model_path=os.path.join(path,'mesh.ply')
        model_3d=load_ply_model(model_path)/1000
        return model_3d

    def get_label(self,img_name):
        #return the label info of one image 
        label_info={}
        # data_path=os.path.join(self.path,img_name)
        # last_folder=img_name.split('/')[0]
        index_name=img_name.split('/')[-1].split('.')[0]
        if "_" in index_name:
            index_name=index_name.split('_')[1]

        index_name=int(index_name)
        img=cv2.imread(os.path.join(self.path,img_name))
        h,w,_=img.shape
        labelpath=os.path.join(self.path,'data')
        # rot_name=img_name.replace('color','rot').replace('.jpg','.rot')
        # tra_name=img_name.replace('color','tra').replace('.jpg','.tra')
        rot_name='rot'+str(index_name)+'.rot'
        tra_name='tra'+str(index_name)+'.tra'
        pose=read_pose(os.path.join(labelpath,rot_name),os.path.join(labelpath,tra_name))
        model=self.model
        model_pro=project('linemod',pose,model)
        #get segmentation
        segm=get_object_seg(model_pro,[h,w])
        #use segm to get the area and bbox
        rle=ann2RLE(segm,[h,w])
        _area=area(rle)
        bbox=ann2box(rle)
        #2d keypoints
        # corner_3d=np.loadtxt(os.path.join(self.path,'corner.txt'))/1000
        fps_3d=np.loadtxt(os.path.join(self.path,'fps_n.txt')) # already model devided by 1000
        # can give corner_3d or fps_n to kp_3d to become keypoints to train 
        kp_3d=fps_3d
        kp_2d=project('linemod',pose,kp_3d)
        #extern one dimension for kp_2d about the keypoint is visable or not
        num_keypoints=kp_2d.shape[0]
        extern_d=np.array([2]*num_keypoints).reshape(num_keypoints,1)
        kp_2d_val=np.c_[kp_2d,extern_d]
        #write the label info to label_info
        label_info['model']=model
        label_info['pose']=pose
        label_info['keypoints']=kp_2d_val.flatten().tolist()
        label_info['segmentation']=segm
        label_info['area']=float(_area[0])
        label_info['bbox']=bbox[0].tolist()
        # print(type(segm))
        # print(type(float(_area[0])))
        # print(type(bbox))
        #
        label={}
        label['1']=label_info
        return label
    



        
        
        
# Linemod('F:\data\linemod_dataset','cat')