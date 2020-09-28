import os

from coco_tools import CocoFormat
from linemod_tools import Linemod,LabelInfoGenerateData
def main(root_path,class_name,labelname):
    #get coco format of object train label 
    '''
    root_path: the linemod dataset root folder
    class_name:the object name
    '''
    #initialize 
    dataset=Linemod(root_path,class_name)

    #load name txt file
    obj_path=root_path+'/'+class_name
    labeltxt=os.path.join(obj_path,labelname)
    print("coco initialize ...")
    ob_coco=CocoFormat('linemod',obj_path)
    ob_coco.categories_content(dataset)
    #
    print("process...")
    #process all image
    with open(labeltxt,'r') as fp:
        img_list=fp.readlines()
        fp.close()
        for i,per_name in enumerate(img_list):
            id=i
            
            per_name=per_name.replace('\n','')
            # print(per_name)
            img_name=per_name.split('/')[-2]+'/'+per_name.split('/')[-1]
            print(img_name)
            label=LabelInfoGenerateData(obj_path,img_name)
            
            ob_coco.getimg(os.path.join(obj_path,img_name),img_name.split('/')[-1],id)
            ob_coco.getann_per_img(id,label)
            # print(ob_coco.ann_id)

    print(i)
    ob_coco.save_json(obj_path)

    return True


main('/home/swh/pose_estimation/dataset_process/dataset_generator/LINEMOD','ape','train.txt')
