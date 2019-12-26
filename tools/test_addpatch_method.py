from mmdet.apis import init_detector, inference_detector, show_result
import os
import argparse
import sys
import torch
# print('config '/media/wrc/0EB90E450EB90E45/mmdetection/configs/guided_anchoring/ga_faster_r50_caffe_fpn_1x.py' checkpoint '/media/wrc/0EB90E450EB90E45/mmdetection/work_dirs/ga_faster_rcnn_r50_caffe_full_model_fpn_1x/latest.pth' ')
def parse_args():
    parser = argparse.ArgumentParser(description='in and out imgs')
    parser.add_argument('--config', dest='config',help='config_file',default='/media/wrc/0EB90E450EB90E45/mmdetection/configs/guided_anchoring/ga_faster_r50_caffe_fpn_1x.py', type=str)
    # parser.add_argument('--config', help='config_file',default='/media/wrc/0EB90E450EB90E45/mmdetection/configs/guided_anchoring/ga_faster_r50_caffe_fpn_1x.py', type=str)
    parser.add_argument('--checkpoint', dest='checkpoint',help='checkpoint_file',default='/media/wrc/0EB90E450EB90E45/mmdetection/work_dirs/ga_faster_rcnn_r50_caffe_full_model_fpn_1x/latest.pth', type=str)
    # parser.add_argument('--checkpoint', help='checkpoint_file',default='/media/wrc/0EB90E450EB90E45/mmdetection/work_dirs/ga_faster_rcnn_r50_caffe_full_model_fpn_1x/latest.pth', type=str)

    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

def main():
  args = parse_args()
  config_file = args.config
  checkpoint_file = args.checkpoint
  model = init_detector(config_file, checkpoint_file)
  print(model.CLASSES)
  root ='/media/wrc/0EB90E450EB90E45/mmdetection/needShow'
  savedir = '/media/wrc/0EB90E450EB90E45/mmdetection/save_dir/'
  img = '/media/wrc/0EB90E450EB90E45/mmdetection/needShow/000099.png'
  result = inference_detector(model,img)
  savename = savedir + 'pic.png'
  show_result(os.path.join(root,img), result, model.CLASSES,out_file=savename)

if __name__ == '__main__':
    main()