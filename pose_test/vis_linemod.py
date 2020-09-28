# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from refinenet.code.linemod_tools import LabelInfo


import time
import os 
import json
import numpy as np
import glob
from evaluation.evaluation import *
from visualization import *
def main(val_path):
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="../configs/caffe2/keypoints_R_101_FPN.yaml",
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

    #
    results=[]
    print("testing ...")
    
    # val_path='./../datasets/linemod/ape_train'
    # val_list=glob.glob(val_path+'/*.png')
    # val_label=os.path.join(val_path,'val.txt')

    #metric
    add=0
    # adds=0
    p_err=0
    # ps_err=0

    val_label=os.path.join(val_path,'val.txt')
    # print(val_label)
    obj_name=val_path.split('/')[-1]
    root_path=val_path
    tmp_path=val_path
    name=obj_name+'_train'
    val_path=val_path.replace(obj_name,'data')
    val_rootpath=os.path.join(val_path,name)

    K=np.array([[572.4114, 0., 325.2611],
                [0., 573.57043, 242.04899],
                [0., 0., 1.]])

    with open('./distance/'+obj_name+'.txt','r') as f:
        diameter=float(f.readline())/100.
        f.close()

    # print(diameter)
    length=0
    with open(val_label,'r') as fp:

        val_imgs=fp.readlines()

        # print(val_imgs)

        fp.close()

        for imgname in val_imgs[:]:
            per_dict={}

            imgname=imgname.replace('\n','').split('/')[-1]
            print(imgname)
            name=int(imgname.split('.')[0])
            ori_name='color'+str(name)+'.jpg'
            imgpath=os.path.join(val_rootpath,imgname)
            # imgpath=os.path.join(root_path,'JPEGImages',imgname)
            # print(imgpath)
            img=cv2.imread(imgpath)
            # print(img)
            try:
                _,box,score,kpts= coco_demo.run_on_opencv_image(img)
                np_box=box.cpu().numpy()
                np_score=score.cpu().numpy()
                p2d=kpts[0,:,:2]
            except:
                continue
            length+=1
            label_rootpath=tmp_path.replace(obj_name,'LINEMOD')
            label_path=os.path.join(label_rootpath,obj_name)
            label=LabelInfo(label_path,ori_name)
            model_3d=label.model
            pose_gt=label.pose
            p3d=np.array(label.fps["fps_8"])
            pose_pred=pnp(p3d,p2d,K)
            box=get_obb(model_3d)
            bb2d_pred=project_K(box,pose_pred,K)
            bb2d_gt=project_K(box,pose_gt,K)
            ab_path=os.path.dirname(os.path.abspath(__file__))
            if not os.path.exists(os.path.join(ab_path,'LINEMOD')):
                os.mkdir(os.path.join(ab_path,'LINEMOD'))
            
            if not os.path.exists(os.path.join(ab_path,'LINEMOD',obj_name)):
                os.mkdir(os.path.join(ab_path,'LINEMOD',obj_name))
            save_path=os.path.join(os.path.join(ab_path,'LINEMOD',obj_name),imgname)
            draw_bbox(img,bb2d_pred[None,None,...],bb2d_gt[None,None,...],save_path)



    return True



        




            



    


# if __name__ == "__main__":

    # main('./../datasets/linemod/ape')
    # compute_error()

obj_list=['ape','benchvise','can','cat','driller','duck','glue','holepuncher','iron','phone']
main('/home/whs/pose_estimation/maskrcnn-benchmark-master/datasets/linemod/benchvise')

# root_path='/home/whs/pose_estimation/maskrcnn-benchmark-master/datasets/linemod'
# def test(path,obj_list):
#     for obj in obj_list:
#         obj_path=os.path.join(path,obj)
#         main(obj_path)


#     print("Done!")
#     return True


# test(root_path,obj_list)

