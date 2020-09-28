import evaluation
import os 
import json

# import model
from model import KeypointRCNNFeatureExtractor,KeypointRCNNPredictor2,RefineNet
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from loss import loss as Loss
import cv2
import dataset
import numpy as np
from plyfile import PlyData
from test_utils import evaluation as test

# import sys
# sys.path.append('./..')
# from datasets.code import LabelInfo

from code.linemod_tools import LabelInfo

from code.dataset_utils import get_intrinsic_cam

def main():

    in_channels=8
    num_keypoints=8

    modelpath='./../tools/results/ape_101/refinement/model_regression/4999.pth'
    predfile='./../tools/results/ape_101/refinement/val_8_101_pred.json'

    gtfile='./../datasets/LINEMOD/ape/ape_val_gen.json'

    # model_3d=PlyData.read('./../dataset/LINEMOD/ape/mesh.ply')

    ape_path='./../datasets/LINEMOD/ape'

    #load model 
    K=get_intrinsic_cam('linemod')

    print("RefineNet building...")
    Feature_extractor=KeypointRCNNFeatureExtractor(in_channels)
    Predictor=KeypointRCNNPredictor2(16,num_keypoints)
    model=RefineNet(Feature_extractor,Predictor)
    print("model build done ") 

    model.load_state_dict(torch.load(modelpath))
    model.cuda()
    model.eval()

    with open (predfile,'r') as fp:
        pred_data=json.load(fp)
        fp.close()
    with open(gtfile,'r') as fp:
        gt_data=json.load(fp)
        fp.close()

    images=gt_data["images"]
    # print(images)
    anns =gt_data["annotations"]

    pred_keys=[x for x in pred_data.keys()]

    error=0
    num_count=0
    for imgname in pred_keys:
        data=pred_data[imgname]
        feature=torch.Tensor(np.array(data[1])).cuda() #[b,c,h,w]

        print(feature.shape)
        #test
        output=model(feature)
        
        # offset_x,offset_y=evaluation.get_offset(output)
        offset_x,offset_y=evaluation.offset(output)
        #get coordinate
        offset_x=offset_x.cpu().numpy()
        offset_y=offset_y.cpu().numpy()
        #
        np_pred_kpts=np.squeeze(np.array(data[0]))
        w,h=data[2]

        # o_w,o_h=(640,480)

        
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

        rois_w=orig_w/w
        rois_h=orig_h/h

        refine_coord=evaluation.get_coordinate(np_pred_kpts,[offset_x,offset_y],[orig_w,orig_h,w,h])


        dis_error=np.sqrt((refine_coord[:,0]-gt_kpts[:,0])**2+(refine_coord[:,1]-gt_kpts[:,1])**2)

        # dis_error=np.sqrt((np_pred_kpts[:,0]*rois_w-gt_kpts[:,0])**2+(np_pred_kpts[:,1]*rois_h-gt_kpts[:,1])**2)

        error+=np.sum(dis_error)/num_keypoints

        #evaluation

        print(dis_error)

        name=int(imgname.split('.')[0])
        imgname='color'+str(name)+'.jpg'
        label=LabelInfo(ape_path,imgname)
        pose_gt=label.pose
        model_3d=label.model

        p2d=refine_coord[:,0:2]
        p3d=np.array(label.fps["fps_8"])

        # print(p3d.shape)
        pose_pred=test.pnp(p3d,p2d,K)

        # print(pose_pred)

        diff=test.projection_2d(pose_pred,pose_gt,model_3d,K)

        if diff<1:
            num_count+=1


        

        


        



        


        
        # print(np.sum(dis_error)/num_keypoints)
        # error+=np.sum(dis_error)/num_keypoints

        # print(refine_coord[:,-1])

    print(error/len(pred_keys))

    print(num_count/len(pred_keys))
    return True


def compute_error():
    repath='./../pose_test/result.json'
    labelpath='./../datasets/linemod/annotations/ape_val.json'
    K=get_intrinsic_cam('linemod')
    ape_path='./../datasets/LINEMOD/ape'


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
    num_count=0
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

        name=int(imgname.split('.')[0])
        imgname='color'+str(name)+'.jpg'
        label=LabelInfo(ape_path,imgname)
        pose_gt=label.pose
        model_3d=label.model

        # dist_x=np.power(gt_kpts[:,0]-pre_kpts[:,0],2)

        # dist_y=np.power(gt_kpts[:,1]-pre_kpts[:,1],2)
        p2d=pre_kpts[:,0:2]
        p3d=np.array(label.fps["fps_8"])

        # dist_xy=np.sqrt(dist_x+dist_y)

        pose_pred=test.pnp(p3d,p2d,K)

        # print(pose_pred)

        diff=test.projection_2d(pose_pred,pose_gt,model_3d,K)

        if diff<5:
            num_count+=1

        # print(dist_xy)
        # # print(pre_kpts[:,-1])
        # dist_error+=dist_xy.sum()/len(dist_xy)

        # print(gt_kpts.shape)

    # print(dist_error/len(results))
    print(num_count/len(results))

main()
# compute_error()