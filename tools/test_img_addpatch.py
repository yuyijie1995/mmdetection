import os
import mmcv
import numpy as np
import random as rd
import numpy.random as random
import cv2
import scipy.misc


def bbox_iou( box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2
    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)
    # Intersection area
    inter_width = inter_rect_x2 - inter_rect_x1 + 1
    inter_height = inter_rect_y2 - inter_rect_y1 + 1
    if inter_width > 0 and inter_height > 0:  # strong condition
        inter_area = inter_width * inter_height
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
        iou = inter_area / (b1_area + b2_area - inter_area)
    else:
        iou = 0
    return iou
data_path='/media/wrc/0EB90E450EB90E45/mmdetection/data/coco/train2017/000513.png'
img_old = mmcv.imread(data_path)#BGR格式
img = mmcv.imread(data_path)#BGR格式
from PIL import Image
cv2.imwrite('cv2_write.jpg',img)
#
# img2=Image.open(data_path)
# img2=np.array(img2)
image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

# image.save("Image_test_1.jpg")

scale_range=(0.9,1.0,1.1,1.2,1.3,1.4,1.5)
gt_bboxes=[[567.0, 173.0, 771.0, 365.0], [1162.0, 177.0, 1240.0, 373.0], [718.0, 168.0, 841.0, 225.0], [687.0, 171.0, 761.0, 207.0], [667.0, 173.0, 734.0, 200.0], [507.0, 176.0, 542.0, 200.0], [40.0, 192.0, 229.0, 266.0], [580.0, 172.0, 604.0, 188.0], [350.0, 180.0, 425.0, 215.0], [401.0, 172.0, 451.0, 208.0], [456.0, 174.0, 499.0, 198.0], [513.0, 175.0, 553.0, 192.0], [233.0, 177.0, 311.0, 214.0], [220.0, 178.0, 302.0, 205.0]]
width=1242
height=375

for inx, bbox in enumerate(gt_bboxes):
    # 填充mask，label，bbox

    bbox_width, bbox_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if bbox_width > 30 or bbox_height > 30 or bbox_width < 5 or bbox_height < 5:
        continue
    new_bbox_left = random.randint(0, width - bbox_width)
    new_bbox_top = random.randint(0, height - bbox_height)
    b1 = int(bbox[1]+1)
    b3 = int(bbox[3])
    b0 = int(bbox[0]+1)
    b2 = int(bbox[2])
    bbox_width = b2 - b0
    bbox_height = b3 - b1

    # scale_ratio = rd.sample(scale_range, 1)[0]
    scale_ratio =1.4
    scaled_bbox_width = int(bbox_width * scale_ratio)
    scaled_bbox_height = int(bbox_height * scale_ratio)
    bbox1 = [new_bbox_left, new_bbox_top, new_bbox_left + scaled_bbox_width,
             new_bbox_top + scaled_bbox_height]
    ious = [bbox_iou(bbox1, bbox2) for bbox2 in gt_bboxes]

    if max(ious) <= 0.3:
        # crop = img[b1:b3, b0:b2, :]
        # crop_float = crop.astype(np.float32)
        # scaled_crop = mmcv.imresize(crop, (scaled_bbox_width, scaled_bbox_height))
        # print(scaled_bbox_width,scaled_bbox_height)
        # if scaled_bbox_height<5 or scaled_bbox_width<5:
        region = image.crop((b0, b1, b2 - 1, b3 - 1))
        region = region.resize((scaled_bbox_width, scaled_bbox_height), Image.ANTIALIAS)
        image.paste(region, (new_bbox_left, new_bbox_top))
        image.save("Image_test_paste.jpg")

        img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        print(img)
        # scaled_crop = scipy.misc.imresize(crop, (scaled_bbox_height, scaled_bbox_width))
        # if scaled_bbox_height == scaled_crop.shape[0] and scaled_bbox_width == scaled_crop.shape[1]:
        #     img[new_bbox_top:new_bbox_top + scaled_bbox_height, new_bbox_left:new_bbox_left + scaled_bbox_width,
        #     :] = scaled_crop


        #cv2.imwrite('crop_resize{0}.png'.format(inx),crop)
        #
        cv2.rectangle(img,(bbox1[0],bbox1[1]),(bbox1[2],bbox1[3]),(0,255,0),2)
        cv2.imwrite('test_img{0}_cv2.png'.format(inx),img)