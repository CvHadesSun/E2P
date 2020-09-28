# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

import time
import os 
import json
import numpy as np


def main():
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
    val_rootpath='./../datasets/linemod/ape_train'
    with open('val.txt','r') as fp:

        val_imgs=fp.readlines()

        # print(val_imgs)

        fp.close()

        for imgname in val_imgs:
            per_dict={}
            imgname=imgname.replace('\n','').split('/')[-1]
            imgpath=os.path.join(val_rootpath,imgname)
            print(imgname)
            img=cv2.imread(imgpath)
            # print(img)
            _,box,score,kpts= coco_demo.run_on_opencv_image(img)
            np_box=box.cpu().numpy()
            np_score=score.cpu().numpy()
            per_dict[imgname]=[np_box.tolist(),np_score.tolist(),kpts.tolist()]
            results.append(per_dict)

    # to save results

    json_dict={}
    json_dict["ape"]=results

    with open('./result.json','w') as f:
        json.dump(json_dict,f)
        f.close()


    return True



        
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
        # print(pre_kpts[:,-1])
        dist_error+=dist_xy.sum()/len(dist_xy)

        # print(gt_kpts.shape)

    print(dist_error/len(results))   


            



    


if __name__ == "__main__":
    main()
    compute_error()
