from functools import partial

import mmcv
import numpy as np
from six.moves import map, zip


def tensor2imgs(tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
        img = mmcv.imdenormalize(
            img, mean, std, to_bgr=to_rgb).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs


def multi_apply(func, *args, **kwargs):
    #这个函数就是将多个序列中的每一组元素 都通过func函数，再将所的结果转置后返回。比如有两个列表list1，list2，要计算这两个列表的点加和点成，可以通过一个函数同时返回两个数的和和差
    #比如 lambda x,y:(x+y,x*y)再使用map函数，也就是：map(lambda x,y:(x+y,x*y),list1,list2)，但是这样的结果是按照[(sum,product),(sum,product),...]这样的形式组织的，所以要将
    #他们进行转置，这样才能让结果中的和在一个列表中，差也在一个列表中。
    # 而用到这个函数则涉及到一个设计思想，那就是将问题按照不同的角度去分解。无论是在mmdet / models / anchor_heads / anchor_head.py还是在mmdetection / mmdet / core / anchor / anchor_target中，都能看到很多_single()
    # 结尾的函数，这样的函数解决的就是分解后的一个小问题。而将一个列表中每一个元素经过multi_apply()
    # 函数，再将结果组合起来，就得到了一个大问题的结果。主要分解的角度有两个，一个是图片，另一个是特征图的尺度。这个角度的意思其实就是说在流程进行中，涉及到的数据的第一个维度的含义，第一种是图片数目，
    # 也就是一个batch中图片的数目作为第一个维度；第二种是特征图尺度的数目，在fasterrcnn中有五个特征图，也就是第一个维度等于5。

    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret
