import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from ..registry import NECKS
from ..utils import ConvModule


@NECKS.register_module
class RESFPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(RESFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.downup_sampling=nn.ModuleList()
        self.res_convs=nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                activation=self.activation,
                inplace=False)
            self.lateral_convs.append(l_conv)

        for i in range(self.start_level, self.backbone_end_level+1):

            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            self.fpn_convs.append(fpn_conv)

        for i in range(self.start_level,self.backbone_end_level):
            down_sampling=nn.MaxPool2d(2,stride=2)

            self.downup_sampling.append(down_sampling)
        for i in range(self.start_level,self.backbone_end_level):
            res_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                activation=self.activation,
                inplace=False)
            self.res_convs.append(res_conv)
        self.conv_p6=ConvModule(in_channels[-1],out_channels,3,stride=2,padding=1,
                                conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                                activation=self.activation,
                                inplace=False
                                )



    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        #inputs 是四个ndarray 第一个是(4,256,104,336) 已知输入是 800，1333的尺寸 这里就是stride 8的一个layer  P3-P6对应的resnet输出作inputs
        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        #self.lateral_conv 四个 1*1的卷积 就是把 四个不同的输入特征的通道数【256，512，1024，2048】统一为256
        p6=self.conv_p6(inputs[-1])
        p6_down=F.interpolate(p6,scale_factor=2,mode='nearest')

        # build top-down path
        used_backbone_levels = len(laterals)
        laterals[used_backbone_levels-1]+=p6_down

        for i in range(used_backbone_levels - 1, 0, -1):
            #用最近邻差值实现上采样 从P5 stride 32 layerals[3] 开始采 把laterals[3]上采样两倍后和laterals[2]进行相加操作
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]#就是一组3*3卷积 对融合完的P3-P6特征进行消除魂叠的效果
        outs.append(p6)
        #out0 是P3最大分辨率

        N=[]
        for i in range(used_backbone_levels+1):
            if i==0:
                N.append(outs[i])
            else:
                N.append(self.downup_sampling[i-1](N[i-1])+outs[i])
                # N.append(F.interpolate(N[i-1],scale_factor=0.5,mode='nearest')+outs[i])

        #残差块
        res_out=[]
        for i in range(used_backbone_levels):
            res_out.append(N[i]+self.res_convs[i](inputs[i]))
        res_out.append(N[-1]+p6)
        #再来一组33卷积消除魂叠

        outs2 = [
            self.fpn_convs[i](res_out[i]) for i in range(used_backbone_levels+1)
        ]  # 就是一组3*3卷积 对融合完的P3-P6特征进行消除魂叠的效果
        # if self.num_outs>len(res_out):
        #     if not self.add_extra_convs:
        #         for i in range(self.num_outs-used_backbone_levels):
        #             outs2.append(F.max_pool2d(outs[-1],1,stride=2))
        return tuple(outs2)
