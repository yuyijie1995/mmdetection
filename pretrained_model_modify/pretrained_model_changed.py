import torch
num_classes=4
model_coco=torch.load('/media/wrc/0EB90E450EB90E45/mmdetection/checkpoints/ga_faster_r50_caffe_fpn_1x_classes_4.pth')
model_coco['state_dict']['bbox_head.fc_cls.weight'].resize_(num_classes,1024)
# model_coco['state_dict']['bbox_head.0.fc_cls.weight'].resize_(num_classes,1024)
# model_coco['state_dict']['bbox_head.1.fc_cls.weight'].resize_(num_classes,1024)
# model_coco['state_dict']['bbox_head.2.fc_cls.weight'].resize_(num_classes,1024)

model_coco['state_dict']['bbox_head.fc_cls.bias'].resize_(num_classes)
# model_coco['state_dict']['bbox_head.0.fc_cls.bias'].resize_(num_classes)
# model_coco['state_dict']['bbox_head.1.fc_cls.bias'].resize_(num_classes)
# model_coco['state_dict']['bbox_head.2.fc_cls.bias'].resize_(num_classes)

torch.save(model_coco,"/media/wrc/0EB90E450EB90E45/mmdetection/work_dirs/ga_faster_rcnn_r50_caffe_full_model_fpn_1x/test_classes_%d.pth"%num_classes)
