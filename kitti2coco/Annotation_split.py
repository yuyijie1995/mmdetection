# -*- coding: utf-8 -*-
import os
import glob
import shutil
import pdb
file_cate=['train.txt','val.txt','test.txt']
srt_Annotations_path='/home/wrc/yuyijie/KITTI/VOCdevkit/VOC2007/Annotations/'
drt_Ann_path='/home/wrc/yuyijie/KITTI/VOCdevkit/VOC2007/'

txt_path='/home/wrc/yuyijie/KITTI/VOCdevkit/VOC2007/ImageSets/Main/'
for idx in file_cate:
    ann_path=os.path.join(drt_Ann_path,'Annotation_coco',idx[:-4])
    if not os.path.exists(ann_path):
        os.makedirs(ann_path)
    label_path=os.path.join(txt_path,idx)
    pdb.set_trace()
    label_txt=open(label_path)
    lines=label_txt.readlines()
    for line in lines:
        file_name=line.strip()+'.xml'
        file_path=os.path.join(srt_Annotations_path,file_name)
        ann_path1=ann_path+'/'+file_name
        print(ann_path1)
        shutil.copy(file_path,ann_path1)




    