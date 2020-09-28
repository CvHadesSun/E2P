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
import scipy.io as sio
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main(val_path, obj_id):
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

    val_img_list = glob.glob(val_path + '/test_data/*-color.png')

    # K = np.array([[572.4114, 0., 325.2611],
    #               [0., 573.57043, 242.04899],
    #               [0., 0., 1.]])

    add_v = 0
    rep_v = 0
    length = 0

    # obj = {1: 2,
    #        2: 3,
    #        3: 4,
    #        4: 5,
    #        5: 6,
    #        6: 7,
    #        7: 8,
    #        8: 9,
    #        9: 10,
    #        10: 11,
    #        11: 19,
    #        12:21,
    #        13:24,
    #        14:25,
    #        16:35,
    #        16:36,
    #        17:37,
    #        18:40,
    #        19:51,
    #        20:52,
    #        21:61}
    
    for imgpath in val_img_list[:]:
        # per_dict = {}
        
        img = cv2.imread(imgpath)
        imgname = imgpath.split('/')[-1]
        # print(imgname)
        img = cv2.imread(imgpath)
        try:
            
            labels, box, score, kpts = coco_demo.run_on_opencv_image(img)

            labels_np = labels.cpu().numpy()
            np_kpts = kpts
            ind = np.where(labels_np == obj_id)
        
        # print(labels_np )
            if len(ind[0]) == 0:
                continue
        # print(imgname)
        # print(len(ind[0]))
            # print(imgname)
            obj_kpts = np_kpts[ind[0][0]]  # [8,3]
            add, rep = error_cp(val_path, obj_id, imgname,  obj_kpts)
        
        except:
            continue

        length += 1
        # print(length)
        if add:
            add_v += 1
        if rep:
            rep_v += 1

    print("ADD metric:{}".format(add_v / length))
    print("REP metric:{}".format(rep_v / length))

    return add_v / length,rep_v / length


def compute_error():
    repath = './result.json'
    labelpath = './../datasets/linemod/annotations/ape_val.json'

    with open(repath, 'r') as f1:
        results = json.load(f1)
        f1.close()
    with open(labelpath, 'r') as f2:
        labels = json.load(f2)
        f2.close()
    images = labels["images"]

    anns = labels["annotations"]

    results = results["ape"]
    dist_error = 0
    for item in results:
        # print(item)
        imgname = [x for x in item.keys()][0]
        print(imgname)
        for img in images:
            if img["file_name"] == imgname:
                id = img["id"]
                for ann in anns:
                    if ann["image_id"] == id:
                        gt_box = ann["bbox"]
                        gt_kpts = ann["keypoints"]

        gt_kpts = np.array(gt_kpts).reshape(len(gt_kpts) // 3, 3)

        pre_box, pre_prob, pre_kpts = item[imgname]
        # pre_prob= item.values()[1]
        # pre_kpts=item.values()[2]
        pre_box = pre_box[0]
        pre_kpts = np.array(pre_kpts[0])

        # print()
        # print(pre_kpts)

        dist_x = np.power(gt_kpts[:, 0] - pre_kpts[:, 0], 2)
        dist_y = np.power(gt_kpts[:, 1] - pre_kpts[:, 1], 2)

        dist_xy = np.sqrt(dist_x + dist_y)

        print(dist_xy)
        # print(pre_kpts[:,-1])
        dist_error += dist_xy.sum() / len(dist_xy)

        # print(gt_kpts.shape)

    print(dist_error / len(results))


def error_cp(root_path, obj_id, imgname, kpts, num_kpts=16):
    #
    obj = {1: "002_master_chef_can",
           2: "003_cracker_box",
           3: "004_sugar_box",
           4: "005_tomato_soup_can",
           5: "006_mustard_bottle",
           6: "007_tuna_fish_can",
           7: "008_pudding_box",
           8:"009_gelatin_box",
           9:"010_potted_meat_can",
           10:"011_banana",
           11:"019_pitcher_base",
           12:"021_bleach_cleanser",
           13:"024_bowl",
           14:"025_mug",
           15:"035_power_drill",
           16:"036_wood_block",
           17:"037_scissors",
           18:"040_large_marker",
           19:"051_large_clamp",
           20:"052_extra_large_clamp",
           21:"061_foam_brick"}
    # obj = {2: "002_master_chef_can",
    #        3: "003_cracker_box",
    #        4: "004_sugar_box",
    #        5: "005_tomato_soup_can",
    #        6: "006_mustard_bottle",
    #        7: "007_tuna_fish_can",
    #        8: "008_pudding_box",
    #        9:"009_gelatin_box",
    #        10:"010_potted_meat_can",
    #        11:"011_banana",
    #        19:"019_pitcher_base",
    #        21:"021_bleach_cleanser",
    #        24:"024_bowl",
    #        25:"025_mug",
    #        35:"035_power_drill",
    #        36:"036_wood_block",
    #        37:"037_scissors",
    #        40:"040_large_marker",
    #        51:"051_large_clamp",
    #        52:"052_extra_large_clamp",
    #        61:"061_foam_brick"}

    # print("cp error")
    # print(imgname)
    obj_name = obj[obj_id]
    # info
    pose_gt, model, fps ,K,diameter= get_info(root_path, obj_name,obj_id, imgname)
    # solve pnp
    kpts_3d = fps[num_kpts]
    kpts_2d=kpts[:,:2]
    pose_pred = pnp(kpts_3d, kpts_2d, K)
    # print(pose_pred.shape)
    #
    # with open('./distance/' + obj_name + '.txt', 'r') as f:
    #     diameter = float(f.readline()) / 100.
    #     f.close()

    # compute error
    # print(obj_name)
    symt=['024_bowl','036_wood_block','051_large_clamp','061_foam_brick']
    if obj_name in symt:
        rep, _ = projection_2ds(pose_pred, pose_gt, model, K)  # return True or False
        add, _ = adds_metric(pose_pred, pose_gt, model, diameter)  # True or False
    else:
        rep,_ = projection_2d(pose_pred, pose_gt, model, K)  # return True or False
        add, _ = add_metric(pose_pred, pose_gt, model, diameter)  # True or False

    return add, rep


def readply(filepath):
    ply = PlyData.read(filepath)
    data = ply.elements[0].data

    x = data['x']
    y = data['y']
    z = data['z']

    return np.stack([x, y, z], axis=-1)


def readObjFile(file_path):
    #read 3d model file .obj file 
    #return the vertices of 3d model(Nx3):
    #
    vertex=[]
    fp=open(file_path,'r')

    line=fp.readline()

    line=line.rstrip('\n').rstrip('\r')
    i=1
    while True:
        line=str(line)
        if line.startswith('v') and not(line.startswith('vt')) and not(line.startswith('vn')):
            line=line.split()
            # if len(line)==4:
            vertex.append([line[1],line[2],line[3]])
            # i+=1
            # print(line,i)
                

        elif line.startswith('vn') or line.startswith('vt') or line.startswith('s') \
            or line.startswith('f') or line.startswith('usemtl'):
            break
        else:
            line=fp.readline()
            line=line.rstrip('\n').rstrip('\r')
    # print(vertex)
    fp.close()
    vertex=np.array(vertex,dtype=np.float)

    return vertex

def get_info(root_path, obj_name,obj_id, imgname):
    modelpath = os.path.join(root_path, 'models', obj_name)
    # name = int(imgname.split("_")[-1].split('.')[0])
    label_name=imgname.replace('color.png','meta.mat')
    # print(label_name)
    ann=sio.loadmat(os.path.join(root_path,'test_data',label_name))
    # print(ann['cls_indexes'].flatten().tolist(),obj_id)
    id=ann['cls_indexes'].flatten().tolist().index(obj_id)
    


    # gt pose
    # posename = "pose" + str(name) + '.npy'

    # posepath = os.path.join(root_path, "blender_poses", obj_name)
    # pose_gt = np.load(os.path.join(posepath, posename))
    pose_gt=ann['poses'][:,:,id]
    K=ann['intrinsic_matrix']

    # model
    modelname = 'textured.obj'
    model = readObjFile(os.path.join(modelpath, modelname))
    with open(os.path.join(modelpath,'diameter.txt'),'r') as f:
        diameter=float(f.readline())
        # print(diameter)
        f.close() 

    # kpts:
    fps_8 = np.loadtxt(os.path.join(modelpath, 'fps_8.txt'))
    fps_12 = np.loadtxt(os.path.join(modelpath, 'fps_12.txt'))
    fps_16 = np.loadtxt(os.path.join(modelpath, 'fps_16.txt'))

    fps = {8: fps_8, 12: fps_12, 16: fps_16}

    return pose_gt, model, fps,K,diameter




# for i in range(1,8):
#     print("{}:".format(i))
#     main('/home/whs/pose_estimation/maskrcnn-benchmark-master/datasets/occluded_linemod/data', i)


# main('/home/whs/pose_estimation/maskrcnn-benchmark-master/datasets/occluded_linemod/data', 6)
# main('/home/whs/pose_estimation/maskrcnn-benchmark-master/datasets/ycb_video/data',21)



#main
# obj = {1: "002_master_chef_can",
#         2: "003_cracker_box",
#         3: "004_sugar_box",
#         4: "005_tomato_soup_can",
#         5: "006_mustard_bottle",
#         6: "007_tuna_fish_can",
#         7: "008_pudding_box",
#         8:"009_gelatin_box",
#         9:"010_potted_meat_can",
#         10:"011_banana",
#         11:"019_pitcher_base",
#         12:"021_bleach_cleanser",
#         13:"024_bowl",
#         14:"025_mug",
#         15:"035_power_drill",
#         16:"036_wood_block",
#         17:"037_scissors",
#         18:"040_large_marker",
#         19:"051_large_clamp",
#         20:"052_extra_large_clamp",
#         21:"061_foam_brick"}

# root_path='/home/whs/pose_estimation/maskrcnn-benchmark-master/datasets/ycb_video/data'
# for i in range(13,22):
#     print(obj[i])
#     _,_=main(root_path,i)