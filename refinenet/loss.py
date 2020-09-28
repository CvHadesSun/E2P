from torch.nn import functional as F
import torch
import numpy as np
from torch import nn
# num_keypoints=8

def loss(input,target):
    # input [b,c,h,w]
    # target [b,c,3]

    R=1 #per grid height
    device=input.device
    N,C,H,W=input.shape
    valid=target[:,:,-1].cpu().numpy().reshape(N*C) #[b,c,1]
    gt_offset_x=target[:,:,0].reshape(N*C)
    gt_offset_y=target[:,:,1].reshape(N*C)

    
    #process boundary
    indx=(gt_offset_x/R+W/2)<0
    gt_offset_x[indx]=-W/2*R

    indx=(gt_offset_x/R-W/2)>0
    gt_offset_x[indx]=(W/2-1)*R

    indy=(gt_offset_y/R+W/2)<0
    gt_offset_y[indy]=-W/2*R

    indy=(gt_offset_y/R-W/2)>0
    gt_offset_y[indy]=(W/2-1)*R

    #process gt label

    location_x=torch.ceil(gt_offset_x/R+W/2) #[b,c,1]
    location_y=torch.ceil(gt_offset_y/R+H/2) #[b,c,1]

    location_y=location_y-1

    # print(ind.shape)

    ind=location_y<0
    location_y[ind]=0

    location_xy=location_y*W+location_x

    # index=np.array([i for i in range(N*C)],dtype=int)

    # gt_target=torch.zeros(N,C,H,W).to(device)
    # gt_target=gt_target.view(N*C,H*W)

    gt_target=location_xy
    gt_target=gt_target.long()


    # print(location_xy)

    

    input=input.view(N*C,H*W)

    valid_index=np.where(valid>0)
    # print(gt_target[valid_index].shape)

    loss=F.cross_entropy(input[valid_index],gt_target[valid_index])
    # print(loss)

    return loss




    
def loss_l2(input,target):
    #l2 loss

    #target [N,num_keypoints,3]

    #input [N,2*num_keypoints]

    N,C=input.shape

    num_keypoints=C //2

    gt_offset_x=target[:,:,0]  #[N,num_keypoints]
    gt_offset_y=target[:,:,1]


    input=input.reshape(N,2,num_keypoints)

    input_x=input[:,0,:]

    input_y=input[:,1,:]

    input=torch.cat((input_x,input_y),0)

    # print(input)
    target_xy=torch.cat((gt_offset_x,gt_offset_y),0)

    # print(input)

    loss_fn = nn.MSELoss(reduce=True, size_average=True)

    loss=loss_fn(input,target_xy)

    print(target_xy-input)
    
    
    return loss

    










    # F.cross_entropy(keypoint_logits[valid], keypoint_targets[valid])
