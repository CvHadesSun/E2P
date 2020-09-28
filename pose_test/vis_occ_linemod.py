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
from plyfile import PlyData
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

    print("testing ...")

    # val_path='./../datasets/linemod/ape_train'
    # val_list=glob.glob(val_path+'/*.png')
    # val_label=os.path.join(val_path,'val.txt')

    # metric

    val_img_list = glob.glob(val_path + '/test_data/*.png')

    K = np.array([[572.4114, 0., 325.2611],
                  [0., 573.57043, 242.04899],
                  [0., 0., 1.]])

    # add_v = 0
    # rep_v = 0
    # length = 0
    ab_path=os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(ab_path,'Occ-LINEMOD')):
        os.mkdir(os.path.join(ab_path,'Occ-LINEMOD'))

    for imgpath in val_img_list[:]:
        per_dict = {}
        img = cv2.imread(imgpath)
        imgname = imgpath.split('/')[-1]

        img = cv2.imread(imgpath)
        try:
            # print(imgname)
        
            labels, box, score, kpts = coco_demo.run_on_opencv_image(img)

            labels_np = labels.cpu().numpy()
            np_kpts = kpts
            bb2d_pred=[]
            bb2d_gt=[]
            print(imgname)
            for obj_id in range(1,8):
                ind = np.where(labels_np == obj_id)

                try:
                    obj_kpts = np_kpts[ind[0][0]]  # [8,3]
                except:
                    continue
                pose_gt,model,K,pose_pred = get_data(val_path, obj_id, imgname, K, obj_kpts)
                box=get_obb(model)
                b2d_pred=project_K(box,pose_pred,K)
                b2d_gt=project_K(box,pose_gt,K)
                bb2d_pred.append(b2d_pred)
                bb2d_gt.append(b2d_gt)

            #vis
            bb82d_pred=np.array(bb2d_pred)
            bb82d_gt=np.array(bb2d_gt)
            draw_multi_bbox(img,bb82d_pred[:,None,...],bb82d_gt[:,None,...],os.path.join(ab_path,'Occ-LINEMOD',imgname))

        except:
            continue

    return True


def get_data(root_path, obj_id, imgname, K, kpts, num_kpts=12):
    #
    obj = {1: "ape",
           2: "can",
           3: "cat",
           4: "driller",
           5: "duck",
           6: "glue",
           7: "holepuncher"}

    obj_name = obj[obj_id]
    # info
    pose_gt, model, fps = get_info(root_path, obj_name, imgname)
    # solve pnp
    kpts_3d = fps[num_kpts]
    kpts_2d=kpts[:,:2]
    # print(kpts_3d)
    pose_pred = pnp(kpts_3d, kpts_2d, K)
    #
    # with open('./distance/' + obj_name + '.txt', 'r') as f:
    #     diameter = float(f.readline()) / 100.
    #     f.close()

    # # compute error
    # # print(obj_name)
    # if obj_name=='glue':
    #     rep, _ = projection_2ds(pose_pred, pose_gt, model, K)  # return True or False
    #     add, _ = adds_metric(pose_pred, pose_gt, model, diameter)  # True or False
    # else:
    #     rep,_ = projection_2d(pose_pred, pose_gt, model, K)  # return True or False
    #     add, _ = add_metric(pose_pred, pose_gt, model, diameter)  # True or False

    return pose_gt,model,K,pose_pred


def readply(filepath):
    ply = PlyData.read(filepath)
    data = ply.elements[0].data

    x = data['x']
    y = data['y']
    z = data['z']

    return np.stack([x, y, z], axis=-1)


def get_info(root_path, obj_name, imgname):
    modelpath = os.path.join(root_path, 'models', obj_name)
    name = int(imgname.split("_")[-1].split('.')[0])
    # gt pose
    posename = "pose" + str(name) + '.npy'
    posepath = os.path.join(root_path, "blender_poses", obj_name)
    pose_gt = np.load(os.path.join(posepath, posename))

    # model
    modelname = obj_name + '.ply'
    model = readply(os.path.join(modelpath, modelname))

    # kpts:
    fps_8 = np.loadtxt(os.path.join(modelpath, 'fps_8.txt'))
    fps_12 = np.loadtxt(os.path.join(modelpath, 'fps_12.txt'))
    fps_16 = np.loadtxt(os.path.join(modelpath, 'fps_16.txt'))

    fps = {8: fps_8, 12: fps_12, 16: fps_16}

    return pose_gt, model, fps



for i in range(1,8):
    print("{}:".format(i))
    main('/home/whs/pose_estimation/maskrcnn-benchmark-master/datasets/occluded_linemod/data',)


# main('/home/whs/pose_estimation/maskrcnn-benchmark-master/datasets/occluded_linemod/data', 6)
