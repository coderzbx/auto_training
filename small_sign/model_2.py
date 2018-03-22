# -*-coding:utf-8-*-

import sys
# sys.path.insert(0, "/opt/t_segnet_server/code/kd_segnet_server/models/mask_rcnn")
# sys.path.insert(0, "/opt/mx-maskrcnn/incubator-mxnet/python")

import os
import copy
import argparse
import time

from rcnn.symbol import *
from rcnn.utils.load_model import load_param
from rcnn.core.module import MutableModule
from rcnn.processing.bbox_transform import nonlinear_pred, clip_boxes
from rcnn.processing.nms import py_nms_wrapper
from rcnn.io.image import transform
from rcnn.config import generate_config
from labels import kd_traffic_sign_labels
import numpy as np
import cv2

bbox_pred = nonlinear_pred


class ModelMaskrcnn(object):
    class OneDataBatch():
        def __init__(self, img_ori):
            # transform the img into: (1, 3, H, W) (RGB order)
            img = transform(img_ori, (118.366, 112.797, 106.641))
            im_info = mx.nd.array([[img.shape[2], img.shape[3], 1.0]])
            self.data = [mx.nd.array(img), im_info]
            self.label = None
            self.provide_label = None
            self.provide_data = [("data", (1, 3, img.shape[2], img.shape[3])), ("im_info", (1, 3))]

    def __init__(self, gpu_id):
        """
          param mode:
              "instace" : 1000 * (cls_id) + instance_id
              "class"   : cls_id
              "pixel"   : 0/1
        """
        generate_config("resnet_fpn", "KdTrafficSign")
        ctx = [mx.gpu(gpu_id)]

        cur_path = os.path.realpath(__file__)
        cur_dir = os.path.dirname(cur_path)
        self.prefix = cur_dir + "/final"
        self.epoch = 0
        self.thresh = 0.7
        self.nms_thresh = 0.2
        self.use_global_nms = True

        self.class_id = [0, 1, 2, 3, 4, 5, 6]
        self.CLASSES = ('__background__', 'warning', 'prohibition', 'instructive', 'highway', 'directing', 'other')
        self.NUM_CLASSES = 7
        self.NUM_ANCHORS = 3
        sym = get_resnet_fpn_mask_test(num_classes=self.NUM_CLASSES, num_anchors=self.NUM_ANCHORS)

        # Load model
        arg_params, aux_params = load_param(self.prefix, self.epoch, convert=False, ctx=ctx, process=True)

        max_image_shape = (1, 3, 2048, 2448)
        max_data_shapes = [("data", max_image_shape), ("im_info", (1, 3))]

        self.height = max_image_shape[2]
        self.width = max_image_shape[3]

        # self.mod = mx.mod.Module(symbol=sym, context=ctx, data_names=["data", "im_info"])
        self.mod = MutableModule(symbol=sym, data_names=["data", "im_info"], label_names=None,
                                 max_data_shapes=max_data_shapes,
                                 context=ctx)
        self.mod.bind(data_shapes=max_data_shapes, label_shapes=None, for_training=False, force_rebind=True)
        self.mod.init_params(arg_params=arg_params, aux_params=aux_params)

        # nms tool
        self.nms = py_nms_wrapper(self.nms_thresh)

        # result mode
        self.mode = "instance"

        # color info
        self.class_colors = [None] * self.NUM_CLASSES
        for label in kd_traffic_sign_labels:
            if not label.trainId in self.class_id:
                continue
            self.class_colors[label.trainId] = (label.color[2], label.color[1], label.color[0])

        # _start = time.time()
        # # image_path = "/data/deeplearning/dataset/518_20180112142933108/518_20180112143129182390.jpg"
        # image_path = "/data/deeplearning/dataset/534_20180223134029926965.jpg"
        # with open(image_path, "rb") as f:
        #     image_data = f.read()
        # self.do(image_data=image_data, dest_file="/data/deeplearning/dataset/534_20180223134029926965.png")
        # _end = time.time()
        # print("Processed in {} ms".format(str((_end - _start) * 1000)))
        # print("got")

    def check_valid(self, bbox):
        if bbox[2] == bbox[0] or bbox[3] == bbox[1] or bbox[0] == bbox[1] or bbox[2] == bbox[3]:
            return False
        # The box is on the floor
        if bbox[3] >= self.height - 1 or bbox[1] >= self.height - 1:
            return False
        return True

    def global_nms_bbox(self, all_boxes, all_masks):
        expand_all_boxes = []
        for cls in self.CLASSES:
            if "__background__" == cls:
                continue
            cls_ind = self.CLASSES.index(cls)
            cls_boxes = all_boxes[cls_ind]
            expand_all_boxes.append(np.hstack((cls_boxes,
                                               np.tile(cls_ind, (cls_boxes.shape[0], 1)).astype(np.int))))
        all_boxes_set = np.concatenate(expand_all_boxes, axis=0)
        all_masks_set = np.concatenate(all_masks[1:], axis=0)
        all_keep = self.nms(all_boxes_set[:, :-1])
        all_keep_boxes = all_boxes_set[all_keep, :].astype(np.float32)
        all_keep_masks = all_masks_set[all_keep, :].astype(np.float32)
        for cls in self.CLASSES:
            if "__background__" == cls:
                continue
            cls_ind = self.CLASSES.index(cls)
            keep = np.where(all_keep_boxes[:, -1] == cls_ind)[0]
            all_boxes[cls_ind] = all_keep_boxes[keep, :-1]
            all_masks[cls_ind] = all_keep_masks[keep]

    def do(self, image_data, dest_file=None):
        pred_data = None
        try:
            image = np.asarray(bytearray(image_data), dtype="uint8")
            img_ori = cv2.imdecode(image, cv2.IMREAD_COLOR)

            width = img_ori.shape[1]
            height = img_ori.shape[0]

            batch = self.OneDataBatch(img_ori)

            self.mod.forward(batch, False)
            results = self.mod.get_outputs()
            output = dict(zip(self.mod.output_names, results))
            rois = output['rois_output'].asnumpy()[:, 1:]
            scores = output['cls_prob_reshape_output'].asnumpy()[0]
            bbox_deltas = output['bbox_pred_reshape_output'].asnumpy()[0]
            mask_output = output['mask_prob_output'].asnumpy()

            pred_boxes = bbox_pred(rois, bbox_deltas)
            pred_boxes = clip_boxes(pred_boxes, [img_ori.shape[0], img_ori.shape[1]])
            boxes = pred_boxes

            all_boxes = [None for _ in xrange(len(self.CLASSES))]
            all_masks = [None for _ in xrange(len(self.CLASSES))]
            label = np.argmax(scores, axis=1)
            label = label[:, np.newaxis]

            for cls in self.CLASSES:
                cls_ind = self.CLASSES.index(cls)
                cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
                cls_masks = mask_output[:, cls_ind, :, :]
                cls_scores = scores[:, cls_ind, np.newaxis]
                keep = np.where((cls_scores >= self.thresh) & (label == cls_ind))[0]
                cls_masks = cls_masks[keep, :, :]
                dets = np.hstack((cls_boxes, cls_scores)).astype(np.float32)[keep, :]
                keep = self.nms(dets)
                all_boxes[cls_ind] = dets[keep, :]
                all_masks[cls_ind] = cls_masks[keep, :, :]

            # apply global nms
            if self.use_global_nms:
                self.global_nms_bbox(all_boxes, all_masks)

            boxes_this_image = [[]] + [all_boxes[j] for j in range(1, len(self.CLASSES))]
            masks_this_image = [[]] + [all_masks[j] for j in range(1, len(self.CLASSES))]

            scale = 1.0
            im = copy.copy(img_ori)
            h, w, _ = im.shape
            instance_id_img = np.zeros((height, width)).astype(np.int32)
            # instance_id: (bbox, score)
            instance_bbox = {}
            for j, name in enumerate(self.CLASSES):
                if name == '__background__':
                    continue
                dets = boxes_this_image[j]
                masks = masks_this_image[j]
                for i in range(len(dets)):
                    bbox = dets[i, :4] * scale
                    if not self.check_valid(bbox):
                        continue
                    instance_id = j * 1000 + (i + 1)
                    score = dets[i, -1]
                    bbox = map(int, bbox)
                    mask = masks[i, :, :]
                    mask = cv2.resize(mask, (bbox[2] - bbox[0], (bbox[3] - bbox[1])),
                                      interpolation=cv2.INTER_LINEAR)
                    mask[mask > 0.5] = instance_id
                    mask[mask <= 0.5] = 0
                    instance_id_img[bbox[1]: bbox[3], bbox[0]: bbox[2]] = mask.astype(np.int32)
                    instance_bbox[instance_id] = (bbox, score)

            blank_image = np.zeros((height, width, 3), np.uint8)
            for instance_id in instance_bbox:
                cls_id = instance_id / 1000
                cls_color = self.class_colors[cls_id]
                mask_inds = (instance_id_img == instance_id)
                target = blank_image[mask_inds].astype(np.int32) + cls_color
                target[target >= 255] = 255
                blank_image[mask_inds] = target.astype(np.uint8)

            img_array = cv2.imencode('.png', blank_image)
            img_data = img_array[1]
            pred_data = img_data.tostring()

            if dest_file is not None:
                with open(dest_file, "wb") as f:
                    f.write(pred_data)

            return pred_data
        except Exception as e:
            print("recognition error:{}".format(repr(e)))
        finally:
            return pred_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    args = parser.parse_args()

    image_dir = args.dir
    model_net = ModelMaskrcnn(gpu_id=0)

    proc_list = []
    file_list = os.listdir(image_dir)
    for id_ in file_list:
        if not id_.endswith("jpg"):
            continue
        proc_list.append(id_)

    for id_ in proc_list:
        file_path = os.path.join(image_dir, id_)
        dest_dir = os.path.join(image_dir, "mask_rcnn")
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        name_list = str(id_).split(".")
        name_only = name_list[0]

        file_id = name_only + ".png"
        dest_path = os.path.join(dest_dir, file_id)

        # if os.path.exists(dest_path):
        #     continue

        try:
            start = time.time()
            with open(file_path, "rb") as f:
                img = f.read()
                model_net.do(image_data=img, dest_file=dest_path)
            end = time.time()
            print("Processed {} in {} ms".format(dest_path, str((end - start) * 1000)))
        except Exception as e:
            print (repr(e))


if __name__ == '__main__':
    main()
