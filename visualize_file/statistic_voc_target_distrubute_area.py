import os
import matplotlib.pyplot as plt

from PIL import Image
# VEDAI 图像存储位置
src_img_dir = "/media/wrc/0EB90E450EB90E45/data/kitti/train_final/VOC2007/JPEGImages"
# VEDAI 图像的 ground truth 的 txt 文件存放位置
src_txt_dir = "/media/wrc/0EB90E450EB90E45/data/kitti/label_2"
# src_xml_dir = "/media/wrc/0EB90E450EB90E45/data/kitti/Annotations"

img_names = os.listdir('/media/wrc/0EB90E450EB90E45/data/kitti/label_2')
center_x=[]
center_y=[]
max_height=0
max_width=0
for img in img_names:
    img = img[:-4]
    im = Image.open((src_img_dir + '/' + img + '.png'))
    width, height = im.size
    if width>max_width:
        max_width=width
    if height>max_height:
        max_height=height

    # open the crospronding txt file
    gt = open(src_txt_dir + '/' + img + '.txt').read().splitlines()
    # gt = open(src_txt_dir + '/gt_' + img + '.txt').read().splitlines()

    # write in xml file
    # os.mknod(src_xml_dir + '/' + img + '.xml')


    # write the region of image on xml file
    for img_each_label in gt:

        spt = img_each_label.split(' ')  # 这里如果txt里面是以逗号‘，’隔开的，那么就改为spt = img_each_label.split(',')。
        center_x.append((float(spt[6])-float(spt[4]))/2+float(spt[4]))
        center_y.append((float(spt[7])-float(spt[5]))/2+float(spt[5]))

    #     xml_file.write('    <object>\n')
    #     # spt[0] = 'helmet'
    #     xml_file.write('        <name>' + str(spt[0]) + '</name>\n')
    #     xml_file.write('        <pose>Unspecified</pose>\n')
    #     xml_file.write('        <truncated>0</truncated>\n')
    #     xml_file.write('        <difficult>0</difficult>\n')
    #     xml_file.write('        <occluded>'+str(spt[2])+'</occluded>\n')
    #     xml_file.write('        <bndbox>\n')
    #     xml_file.write('            <xmin>' + str(spt[4]) + '</xmin>\n')
    #     xml_file.write('            <ymin>' + str(spt[5]) + '</ymin>\n')
    #     xml_file.write('            <xmax>' + str(spt[6]) + '</xmax>\n')
    #     xml_file.write('            <ymax>' + str(spt[7]) + '</ymax>\n')
    #     xml_file.write('        </bndbox>\n')
    #     xml_file.write('    </object>\n')
    #
    # xml_file.write('</annotation>')
    print('a file %s done'%img)
print('the data had been load')
figure,ax=plt.subplots()

ax.set_xlim(left=0,right=max_width)
ax.set_ylim(bottom=max_height,top=0)

ax.xaxis.tick_top()#将x坐标标记移到上方
plt.scatter(center_x,center_y,alpha=0.6)
plt.show()

