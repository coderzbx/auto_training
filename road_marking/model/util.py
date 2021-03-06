# -*-coding:utf-8-*-

import os
import numpy as np
from functools import partial
from PIL import Image
from multiprocessing import Pool
import mxnet as mx

from weighted_loss_layer import WeightedOhemSoftmax
from weighted_loss_layer import WeightedOhemSoftmaxProp

from seg_labels import self_server_label


cfg = {}


class KdSeg12Id(object):
    def __init__(self):
        """Initializer."""
        super(KdSeg12Id, self).__init__()
        self._label_2_id = np.full((256,), 255)
        for label in self_server_label:
            if label.id in (-1, 255):
                continue
            self._label_2_id[label.id] = label.categoryId

    def __call__(self, data):
        """Functor"""
        return self._label_2_id[data]


class KdSeg12Id2Color(object):
    def __init__(self):
        super(KdSeg12Id2Color, self).__init__()
        self._cmap = np.zeros((256, 3), dtype=np.uint8)
        for label in self_server_label:
            if label.id in (-1, 255):
                continue
            self._cmap[label.categoryId] = label.color

    def __call__(self, data):
        """Functor"""
        return self._cmap[data]


def load_weights(weights_file):
    assert os.path.exists(weights_file)
    prefix = weights_file.split("_ep-")[0] + "_ep"
    epoch = int(weights_file.split("_ep-")[1].split(".")[0])
    print ("prefix: {}, epoch: {}".format(prefix, epoch))
    network, net_args, net_auxs = mx.model.load_checkpoint(prefix, epoch)
    return network, net_args, net_auxs


def color_scale(scale):
    """ColorScale.
    """

    def _impl(data):
        data = data.astype(np.float32, copy=False)
        data *= np.float32(scale)
        return data

    return _impl


def color_normalize(mean, std=None):
    """ColorNormalize.
    """

    def _impl(data):
        data = data.astype(np.float32, copy=False)
        data -= np.array(mean, dtype=np.float32)
        if std is not None:
            data /= np.array(std, dtype=np.float32)
        return data

    return _impl


def _interp_preds_as_impl(imh, imw, pred):
    """
        interpolate each dimension
    """
    pred = pred.astype(np.single, copy=False)
    input_h, input_w = pred.shape[:2]
    if input_h == imh:
        interp_pred = pred
    else:
        # interp_method = Image.BILINEAR
        interp_method = get_interp_method(input_h, input_w, imw, imh)
        interp_pred = np.array(Image.fromarray(pred).resize((imw, imh), interp_method))
    return interp_pred


def interp_preds_as(net_preds, im_shape, threads=4):
    """
        net_preds : (C, H, W)
        im_shape  : (H, W, C)
    """
    imh, imw = im_shape[:2]
    worker = partial(_interp_preds_as_impl, imh, imw)
    if threads == 1:
        ret = [worker(_) for _ in net_preds]
    else:
        pool = Pool(threads)
        ret = pool.map(worker, net_preds)
        pool.close()
        pool.join()
    return np.array(ret)


def get_interp_method(imh_src, imw_src, imh_dst, imw_dst, default=3):
    """
        get_interp_method
    """
    if not cfg.get('choose_interpolation_method', False):
        return default
    if imh_dst < imh_src and imw_dst < imw_src:
        return Image.ANTIALIAS
    elif imh_dst > imh_src and imw_dst > imw_src:
        return Image.CUBIC
    else:
        return Image.LINEAR