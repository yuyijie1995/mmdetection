# -*- coding: utf-8 -*-
# !/usr/bin/python

# pip install lxml

import sys
import os
import shutil
START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = {"Car": 1, "Cyclist": 2, "Pedestrian": 3}





split_list = ['train', 'val', 'test']


def split(txt_path, img_dir, dst_path):
    for idx in split_list:
        txt_list = os.path.join(txt_path, idx + '.txt')
        dst_img_path=os.path.join(dst_path,idx+'set')
        if not os.path.exists(dst_img_path):
            os.makedirs(dst_img_path)

        list_fp = open(txt_list, 'r')# read train.txt

        for line in list_fp:
            line = line.strip()
            print("Processing %s" % (line))
            img_f=os.path.join(img_dir,line+'.png')
            dst_img_f=os.path.join(dst_img_path,line+'.png')
            shutil.copy(img_f,dst_img_f)




if __name__ == '__main__':
    txt_list = '/media/wrc/0EB90E450EB90E45/data/kitti/ImageSets/Main/'
    src_img_dir = '/media/wrc/0EB90E450EB90E45/data/kitti/image_2/'
    dst_img_dir = '/media/wrc/0EB90E450EB90E45/data/kitti/COCO/'
    split(txt_list, src_img_dir, dst_img_dir)


