# from demo.webcam import main
import torch

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

import time
import os 
import json
import numpy as np


def dataloader(img):
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="../configs/caffe2/e2e_keypoint_rcnn_R_50_FPN_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=224,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
    )

    pred=coco_demo.run_on_opencv_image(img)

    keypoints=pred.get_field("keypoints")

    feature=keypoints.get_field("feature")

    return keypoints.keypoints,feature


        
def compute_error():
    repath='./result.json'
    labelpath='./../datasets/linemod/annotations/ape_val.json' 

    with open(repath,'r') as f1:
        results=json.load(f1)
        f1.close()
    with open(labelpath,'r') as f2:
        labels=json.load(f2)
        f2.close()
    images=labels["images"]

    anns=labels["annotations"]   

    results=results["ape"]
    dist_error=0
    for item in results:
        # print(item)
        imgname=[x for x in item.keys()][0]
        print(imgname)
        for img in images:
            if img["file_name"]==imgname:
                id=img["id"]
                for ann in anns:
                    if ann["image_id"] ==id:
                        gt_box=ann["bbox"]
                        gt_kpts=ann["keypoints"]

        gt_kpts=np.array(gt_kpts).reshape(len(gt_kpts)//3,3)
        
        pre_box,pre_prob,pre_kpts=item[imgname]
        # pre_prob= item.values()[1]
        # pre_kpts=item.values()[2]
        pre_box=pre_box[0]
        pre_kpts=np.array(pre_kpts[0])

        # print()
        # print(pre_kpts)

        dist_x=np.power(gt_kpts[:,0]-pre_kpts[:,0],2)
        dist_y=np.power(gt_kpts[:,1]-pre_kpts[:,1],2)

        dist_xy=np.sqrt(dist_x+dist_y)

        print(dist_xy)
        print(pre_kpts[:,-1])
        dist_error+=dist_xy.sum()/len(dist_xy)

        # print(gt_kpts.shape)

    print(dist_error/len(results))   


            
from torch.utils.data import Dataset

class CoordDataset(Dataset):
    def __init__(self, fiel1,fiel2):
        self.feature,self.label=get_label(fiel1,fiel2)

    def __len__(self):
        return len(self.label)

    def __getitem__(self,idx):
        
        return torch.Tensor(self.feature[idx]),torch.Tensor(self.label[idx])



def get_label(file1,file2):
    #pred_results
    pre_data=0
    gt_data=0
    with open(file1,'r') as f1:
        pre_data=json.load(f1)
        f1.close()
    with open(file2,'r') as f2:
        gt_data=json.load(f2)
        f2.close()  
    
    images=gt_data["images"]
    # print(images)
    anns =gt_data["annotations"]

    pred_keys=[x for x in pre_data.keys()]
    feature_data=[]
    label_data=[]
    for imgname in pred_keys:
        data=pre_data[imgname]
        np_feature=np.array(data[1])
        feature_data.append(np.squeeze(np_feature)) #[c,h,w]
        np_pred_kpts=np.squeeze(np.array(data[0]))
        w,h=data[2]
        # print(imgname)
        # find the ground truth of imgname
        for img in images:
            if img["file_name"]==imgname:
                id=img["id"]
                # print(id)
                orig_w=img["width"]
                orig_h=img["height"]
                for ann in anns:
                    if ann["image_id"] ==id:
                        gt_kpts=np.array(ann["keypoints"]).reshape(len(ann["keypoints"])//3,3)

        ##

        w_rois=w/orig_w
        h_rois=h/orig_h

        #to compute offset
        offset=np.zeros(np_pred_kpts.shape)
        gt_resize_kpts=np.zeros(np_pred_kpts.shape)

        #resize coordinates
        gt_resize_kpts[:,0]=gt_kpts[:,0]*w_rois
        gt_resize_kpts[:,1]=gt_kpts[:,1]*h_rois

        offset[:,0:2]=np_pred_kpts[:,0:2]-gt_resize_kpts[:,0:2]
        offset[:,-1]=gt_kpts[:,-1]

        label_data.append(offset)

    return feature_data,label_data






    







