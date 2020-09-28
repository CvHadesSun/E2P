'''
change the other format to coco format and save the info to a json file
@author:cvhadessun 2019-12-13
'''
import os 
import json
import sys
import glob
import cv2
from linemod_tools import Linemod,LabelInfo
import datetime
class CocoFormat(object):
    '''
    transform the coco dataset format
    '''
    def __init__(self,dataset_name,path):
        #path: dict{"model_path":[...],"data_path":[...]}

        self.licenses=[{"name":dataset_name,"id":1,"url":"219.223.170.43"}]
        self.info= {"author":"cvhadessun"}
        self.categories=[]
        self.images=[]
        self.annoatations=[]
        self.path=path
        self.ann_id=0
        date=datetime.datetime.now()
        self.date=str(date.year)+'-'+str(date.month)+'-'+str(date.day)
        # self.ycb=ycb_v

    def getimg(self,path,img_name,id):
        #append image to images and get the w and h of the image
        path_img=path
        # print(path_img)
        img=cv2.imread(path_img)
        h,w=img.shape[:-1]
        dict_img= {"height": h,
              "flickr_url": "ycb-video",
              "license": 1,
              "id":id,
              "width": w,
              "date_captured":self.date,
              "file_name": img_name}
        self.images.append(dict_img)
      

    def getann_per_img(self,id,label_info):
        #save the label of one image to json file
        # id:the image id 
        # label_info:the annotation info of one image
        
        num_objects=label_info.num_objects
        category_ids=label_info.category_ids
        #from the label_info get the annotation and save to annotations
        for i in range(num_objects):
            #initial
            dict_ann={}
            #get one annotation of one object
            pose=label_info.label['{}'.format(category_ids[i])]['pose']
            model=label_info.label['{}'.format(category_ids[i])]['model']
            segmentation=label_info.label['{}'.format(category_ids[i])]['segmentation']
            keypoints=label_info.label['{}'.format(category_ids[i])]['keypoints']
            category_id=category_ids[i]
            area=label_info.label['{}'.format(category_ids[i])]['area']
            bbox=label_info.label['{}'.format(category_ids[i])]['bbox']
            #append to annotations
            dict_ann["area"]=area
            dict_ann["bbox"]=bbox
            dict_ann["category_id"]=category_id
            dict_ann["keypoints"]=keypoints
            dict_ann["iscrowd"]=0
            dict_ann["num_keypoints"]=len(keypoints)//3
            dict_ann["image_id"]=id
            dict_ann["id"]=self.ann_id+i+1
            # print(dict_ann["id"])
            dict_ann["segmentation"]=segmentation  
            # dict_ann['full_mask']=full_polygon

            self.annoatations.append(dict_ann)
        self.ann_id+=1+i
        # print(self.ann_id)



    def categories_content(self,dataset_ob):
        #to save the categories content for all object
        num_category=dataset_ob.categories_info["num_object"]
        _categories=[]
        for i in range(num_category):
            dict_cate={"supercategory": "object",
                "keypoints": ["1", "2", "3", "4",
                            "5", "6", "7", "8",],
                "skeleton": [[1, 2], [1, 3], [2, 4], [1, 5], [5, 7], [7, 8],
                            [3, 4], [5, 6], [2, 6], [6, 8], [8, 4]]
                        }
            dict_cate["id"]=i+1
            dict_cate["name"]=dataset_ob.categories_info["name"][i]
            _categories.append(dict_cate)
        self.categories=_categories


    def save_json(self,save_path):
        #save data as a .json file
        
        json_dict={
            "licenses":self.licenses,
            "categories":self.categories,
            "images":self.images,
            "annotations":self.annoatations,
            "info":self.info

        }
        # json_dict={
        #     "licenses":self.licenses,
        #     "categories":self.categories,
        #     "images":self.images,
        #     "info":self.info

        # }


        json_path=os.path.join(save_path,'kpts_coco.json')
        with open (json_path,'w') as fp:
            json.dump(json_dict,fp)
            print("Done")
            fp.close()


def main(root_path,class_name):
    #generate the coco format label json file 

    '''
    :root_path:the path of root path of dataset 
    :dataset_name: the path of object name,eg:cat,...

    '''
    #initialize 
    dataset=Linemod(root_path,class_name)
    class_ob_path=os.path.join(root_path,class_name)
    img_path=os.path.join(os.path.join(root_path,class_name),'data')
    # categories_info=dataset.categories_info
    img_list=dataset.img_names #the image names make of img_list
    ob_coco=CocoFormat('linemod',img_path)
    ob_coco.categories_content(dataset)
    #process all image 
    for i,per_name in enumerate(img_list):
        id=i
        # print(per_name)
        # print(ob_coco.ann_id)
        label=LabelInfo(class_ob_path,per_name)
        ob_coco.getimg(os.path.join(class_ob_path,'data'),per_name,id)
        ob_coco.getann_per_img(id,label)
        # print(ob_coco.ann_id)


    ob_coco.save_json(class_ob_path)

    return True


# root_path='./../data'
# class_name='cat'
# flag=main(root_path,class_name)