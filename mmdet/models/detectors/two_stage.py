import torch
import torch.nn as nn

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from .test_mixins import BBoxTestMixin, MaskTestMixin, RPNTestMixin


@DETECTORS.register_module
class TwoStageDetector(BaseDetector, RPNTestMixin, BBoxTestMixin,
                       MaskTestMixin):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 shared_head=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwoStageDetector, self).__init__()
        #上来一波build，这里和之前的build没什么区别，在注册器里面去除对应的type类型，然后返回一个对象，以后就可以肆意调用里面的功能了
         # 这里build了backbone, neck, shared_head, rpn_head,bbox_head,mask_head（如果有）
# 　　　　　  # backbone,主干网络用的啥，譬如resnet50, resnext101之类的
# 　　　　　  # neck 一般是FPN,需要指定很多参数，譬如用哪些feature map,之后会详细说
# 　　　　　  # rpn_head继承了anchor_head，是ｒｐｎ的核心功能
        self.backbone = builder.build_backbone(backbone)
        self.count=0#用于在不同iterations的时候进行操作
        if neck is not None:
            self.neck = builder.build_neck(neck)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)

        if mask_head is not None:
            if mask_roi_extractor is not None:
                self.mask_roi_extractor = builder.build_roi_extractor(
                    mask_roi_extractor)
                self.share_roi_extractor = False
            else:
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.bbox_roi_extractor
            self.mask_head = builder.build_head(mask_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(TwoStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def extract_feat(self, img):#forward backbone 和neck函数
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmedetection/tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).cuda()
        # bbox head
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            outs = outs + (cls_score, bbox_pred)
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            mask_pred = self.mask_head(mask_feats)
            outs = outs + (mask_pred, )
        return outs

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_meta (list[dict]): list of image info dict where each dict has:
                'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)#提取backbone和neck
        # print('after backbone:the feature map is:{0}'.format(x))
        # if self.count%2992==1:
        #     unloader = tf1.ToPILImage()
        #     for stage in range(len(x)):
        #         for batch in range(x[stage].shape[0]):
        #             for channel in range(x[stage].shape[1]):
        #
        #                 image = (x[stage]*255)[batch,channel,:,:].cpu().clone()
        #                 image = image.squeeze(0)
        #                 image = unloader(image)
        #                 fm_dir='./featuremap_save/'
        #                 if not os.path.exists(fm_dir):
        #                     os.makedirs(fm_dir)
        #                 image.save(fm_dir+'batch{}-channel{}.jpg'
        #                            .format(stage, batch, channel))
        self.count+=1
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            # rpn_head在上面__init__函数里面build了,返回了rpn_head对象，
            # 但是因为都继承了nn.Module(实现了__call__)，可以直接用实例名字调用里面的forward函数，从而进行了前向传播
            rpn_outs = self.rpn_head(x)#对于之前neck的每一层都会有rpn输出 所以这里是5份输出
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)#把信息叠加在一起了

            # 这里loss是rpn_head里面的实现了，因为rpn_head继承了anchor_head，
            # 所以用了父类anchor_head实现的loss，其实就是anchor的那一套，返回是一个字典
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            #这里要提取proposals来进行
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            # 得到proposal,把ａｎｃｈｏｒ转化为对应的框的信息，然后ＮＭＳ再取top-N个候选框
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals#第一次回归完成得到的所有bbox

        # assign gts and sample proposals
        # 这边assign和sample的代码就是上面讲的流程里的过程
        #接下来rcnn中其实就不用再生成anchors 而是用上面得到的proposals作为anchor使用 同样的不会使用所有的proposals进行训练，只是选择其中的一部分
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[i],
                                                     gt_bboxes[i],
                                                     gt_bboxes_ignore[i],
                                                     gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)#sampling_results包含了 bboxes pos_inds neg_inds pos_label...等

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])#把bbox转成roi 区别就是给每个框的编码前面家一个所属的图片索引
            #因为一次采样多张图片，一般为2张，变成5维的向量，最后把所有的cat一下，变成(n,5)的tensor
            #同时把4张图片的所有bboxes通过这个bbox2roi也堆叠了起来 把4个(512,5) 得到一个2048，5的张量
            # print('rois is :{0}'.format(rois))
            # print('rois shape is :{0}'.format(rois.shape))
            # TODO: a more flexible way to decide which feature maps to use
            # 这里的x是 5个分辨率大小的特征图 这里只取前4层 最小的那个不取
            #把roi对应的特征图拿出来
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            #这里得到的pred是(512,16)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            # if self.count==1:
            #     bbox_pred_cpu=bbox_pred.cpu().detach()
            #     cls_cpu=cls_score.cpu().detach()
            #     rois_cpu=rois.cpu().detach()
            #     bbox_pred_numpy=np.array(bbox_pred_cpu)
            #     cls_score_numpy=np.array(cls_cpu)
            #     rois_numpy=np.array(rois_cpu)
            #     np.save("./bbox_pred_numpy.npy", bbox_pred_numpy)
            #     np.save("./cls_score_numpy.npy", cls_score_numpy)
            #     np.save("./rois_numpy.npy", rois_numpy)




            #返回的是 labels, label_weights, bbox_targets, bbox_weights的集合 bbox_weights 正样本是1 负样本是0
            bbox_targets = self.bbox_head.get_target(sampling_results,
                                                     gt_bboxes, gt_labels,
                                                     self.train_cfg.rcnn)
            # import pdb
            # pdb.set_trace()
            #直接用sampling_results(包含了pos_gt_bboxes) 里面是4张图片的gt（80，4）
            #这里的bbox_pred则是（2048，16）进入bbox_head.loss之后选择里面的pos_ind对应的数值，---》（80，4）
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            #这里的loss先是进入bbox_head.py中进行pred的挑选 根据cls_score从bbox_pred中挑出对应的坐标
            losses.update(loss_bbox)

        # mask head forward and loss
        if self.with_mask:
            if not self.share_roi_extractor:
                pos_rois = bbox2roi(
                    [res.pos_bboxes for res in sampling_results])
                mask_feats = self.mask_roi_extractor(
                    x[:self.mask_roi_extractor.num_inputs], pos_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
            else:
                pos_inds = []
                device = bbox_feats.device
                for res in sampling_results:
                    pos_inds.append(
                        torch.ones(
                            res.pos_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                    pos_inds.append(
                        torch.zeros(
                            res.neg_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                pos_inds = torch.cat(pos_inds)
                mask_feats = bbox_feats[pos_inds]

            if mask_feats.shape[0] > 0:
                mask_pred = self.mask_head(mask_feats)
                mask_targets = self.mask_head.get_target(
                    sampling_results, gt_masks, self.train_cfg.rcnn)
                pos_labels = torch.cat(
                    [res.pos_gt_labels for res in sampling_results])
                loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                                pos_labels)
                losses.update(loss_mask)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.async_test_rpn(x, img_meta,
                                                      self.test_cfg.rpn)
        else:
            proposal_list = proposals

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_meta,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.simple_test_rpn(x, img_meta,
                                                 self.test_cfg.rpn)
        else:
            proposal_list = proposals

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs), img_metas, proposal_list,
            self.test_cfg.rcnn)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(
                self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results
