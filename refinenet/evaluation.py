
import torch

import numpy as np
def get_offset(output):
    # to recover the output get offset
    #output [b,c,h,w]

    R=1 #resolution
    N,C,H,W=output.shape
    output=output.view(N*C,H*W)

    values,index=torch.max(output,1)

    y_index=index//H
    x_index=index%W

    y_index=y_index+1

    #get offset_xy  
    offset_x=x_index*R-W/2
    offset_y=y_index*R-W/2

    return offset_x,offset_y



def get_coordinate(coord,offset_xy,shape):

    #data format:numpy

    o_w,o_h,w,h=shape

    rois_w=o_w/w

    rois_h=o_h/h

    # coord=data[0]

    offset_x,offset_y=offset_xy

    coord[:,0]=(coord[:,0]+offset_x)*rois_w
    coord[:,1]=(coord[:,1]+offset_y)*rois_h

    #pred confidence
    coord[:,2]=np.exp(-np.sqrt((offset_x*rois_w)**2+(offset_y*rois_h)**2))


    return coord


    
def offset(output):
    b,c=output.shape
    resize_output=output.reshape(b,2,c//2)

    offset_x=resize_output[:,0,:]
    offset_y=resize_output[:,1,:]

    print(offset_x)

    return offset_x,offset_y






    







