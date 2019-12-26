import cv2
import os
import glob
img_dir='/media/wrc/0EB90E450EB90E45/mmdetection/data/coco/test2019'
rex_dir='/media/wrc/0EB90E450EB90E45/mmdetection/res/*'
color=[(0,0,255),(0,255,0),(255,0,0)]
classes=['car','cyclist','pedestrain']
txt_lst=glob.glob(rex_dir)
for item in txt_lst:
    if not os.path.getsize(item):
        continue
    img_path = os.path.join(img_dir, item.split('/')[-1][:-4] + '.png')
    img = cv2.imread(img_path)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    infos=open(item,'r').readlines()
    for info in infos:
        # print(info)
        info=info.strip('\n')
        info=info.split('\t')


        try:
            xmin=int(float(info[4]))
            ymin=int(float(info[5]))
            xmax=int(float(info[6]))
            ymax=int(float(info[7]))
            width = xmax - xmin
            height = ymax - ymin
            cv2.rectangle(img, (xmin, ymin),
                    (xmax,ymax),color[classes.index(info[0])], 3)
            # cv2.putText(drawimg, info[0], (xmin,ymin+20 ), cv2.FONT_HERSHEY_COMPLEX, 5, (0, 255, 0), 1)
            # cv2.imshow('name1',drawimg)
            # cv2.waitKey(0)
            # cv2.destroyWindow("draw_0")
        except IndexError:
            print('%s is something wrong!'%(item))
    cv2.imwrite('./out_img/{0}.png'.format(item.split('/')[-1][:-4]), img)
    print('{0}.png complete!'.format(item.split('/')[-1][:-4]))
    #cv2.imshow('show_once',img)




