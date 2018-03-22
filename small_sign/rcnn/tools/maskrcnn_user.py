import sys
import os
import copy
import random
import logging
import argparse
from rcnn.symbol import *
from rcnn.utils.load_model import load_param
from rcnn.core.module import MutableModule
from rcnn.processing.bbox_transform import nonlinear_pred, clip_boxes
from rcnn.processing.nms import py_nms_wrapper
from rcnn.io.image import transform
from rcnn.config import config, default, generate_config
from rcnn.dataset.kd_traffic_sign_labels import kd_traffic_sign_labels
import numpy as np
import cv2
import matplotlib.pyplot as plt

bbox_pred = nonlinear_pred

class MaskrcnnUser(object):
  class OneDataBatch():
    def __init__(self, img_ori):
      # transform the img into: (1, 3, H, W) (RGB order)
      img = transform(img_ori, (118.366, 112.797, 106.641))
      im_info = mx.nd.array([[img.shape[2], img.shape[3], 1.0]])
      self.data = [mx.nd.array(img), im_info]
      self.label = None
      self.provide_label = None
      self.provide_data = [("data",(1, 3, img.shape[2], img.shape[3])), ("im_info", (1, 3))]

  def __init__(self, prefix, epoch, gpu_id, score_thresh, nms_thresh,
               use_global_nms=True, mode="instance"):
    """
      param mode:
          "instace" : 1000 * (cls_id) + instance_id
          "class"   : cls_id
          "pixel"   : 0/1
    """
    ctx = mx.gpu(gpu_id)
    self.thresh = score_thresh
    self.use_global_nms = use_global_nms

    logging.info("Get network symbol.")
    self.class_id = [0, 1, 2, 3, 4, 5, 6]
    self.CLASSES = ('__background__', 'warning', 'prohibition', 'instructive', 'highway', 'directing', 'other')
    self.NUM_CLASSES = 7
    self.NUM_ANCHORS = 3
    assert self.NUM_CLASSES == len(self.CLASSES), "Diff between NUM_CLASSES and len(CLASSES)"
    sym = get_resnet_fpn_mask_test(num_classes=self.NUM_CLASSES, num_anchors=self.NUM_ANCHORS)

    # Load model
    logging.info("Load weights.")
    arg_params, aux_params = load_param(prefix, epoch, convert=True, ctx=ctx, process=True)

    logging.info("Get mododule.")
    max_image_shape = (1, 3, 2048, 2448)
    max_data_shapes = [("data", max_image_shape), ("im_info", (1, 3))]

    self.height = max_image_shape[2]
    self.width = max_image_shape[3]

    self.mod = MutableModule(symbol = sym, data_names = ["data","im_info"], label_names= None,
                             max_data_shapes = max_data_shapes,
                             context=ctx)
    self.mod.bind(data_shapes=max_data_shapes, label_shapes=None, for_training=False)
    self.mod.init_params(arg_params=arg_params, aux_params=aux_params)

    # nms tool
    logging.info("Get nms tool.")
    self.nms = py_nms_wrapper(nms_thresh)

    # result mode
    self.mode = mode

    # color info
    self.class_colors = [None] * self.NUM_CLASSES
    for label in kd_traffic_sign_labels:
        if not label.trainId in self.class_id:
          continue
        self.class_colors[label.trainId] = label.color

  def run(self, img_path, vis_image_file=None):
    img_ori = cv2.imread(img_path)
    instance_bbox, instance_id_img = self.run_img(img_ori)
    result_img = copy.copy(instance_id_img)
    if self.mode == "instance":
      pass
    elif self.mode == "class":
      for instance_id in instance_bbox:
        pixels_inds = np.where(result_img == instance_id)
        result_img[pixels_inds] = instance_id / 1000
    elif self.mode == "pixel":
      pixel_inds = np.where(result_img != 0)
      result_img[pixel_inds] = 1

    if vis_image_file is not None:
      self.vis_result(img_ori, instance_bbox, instance_id_img, vis_image_file)

    return instance_bbox, result_img

  def run_img(self, img_ori):
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
    boxes= pred_boxes

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

    return self.make_result_img(img_ori, boxes_this_image, masks_this_image)

  def make_result_img(self, img_ori, boxes_this_image, masks_this_image, scale=1.0):
    im = copy.copy(img_ori)
    h, w, _ = im.shape
    instance_id_img = np.zeros((h, w)).astype(np.int32)
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
    return instance_bbox, instance_id_img

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

  def vis_result(self, img_ori, instance_bbox, instance_id_img, vis_image_file):
    im = copy.copy(img_ori)
    file_path = os.path.dirname(vis_image_file)
    if not os.path.exists(file_path):
      os.makedirs(file_path)
    for instance_id in instance_bbox:
      cls_id = instance_id / 1000
      ins_id = instance_id % 1000
      cls_color = self.class_colors[cls_id]
      bbox = instance_bbox[instance_id][0]
      score = instance_bbox[instance_id][1]
      cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=cls_color, thickness=2)
      cv2.putText(im, '%s %.3f' % (self.CLASSES[cls_id], score), (bbox[0], bbox[1] + 10),
                  color=(0, 0, 0), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
      mask_inds = (instance_id_img == instance_id)
      target = im[mask_inds].astype(np.int32) + cls_color
      target[target >= 255] = 255
      im[mask_inds] = target.astype(np.uint8)
    cv2.imwrite(vis_image_file, im)

def main():
  logging.basicConfig(format='%(levelname)s: %(asctime)s: %(filename)s:%(lineno)d * %(thread)d %(message)s',
                      datefmt='%Y%m%d-%H:%M:%S',
                      level=logging.INFO)
  # Init config
  generate_config("resnet_fpn", "KdTrafficSign")

  prefix = "/data/deeplearning/mask-rcnn/output/model/final"
  epoch = 0
  gpu_id = 0
  score_thresh = 0.2
  nms_thresh = 0.2
  maskrcnn_user = MaskrcnnUser(prefix, epoch, gpu_id, score_thresh, nms_thresh,
                               use_global_nms=True, mode="class")

  img_files = [
    "443_20180112143345550548_00_004.jpg",
    "4204_20171226022654800949_00_004.jpg",
    "4186_20171225070815924433_00_004.jpg",
    "424_20180112145024315963_00_004.jpg",
    "446_20180112153205890405_00_004.jpg",
    "574_20180112144144675891_00_004.jpg",
    "424_20180112145025566570_00_004.jpg",
    "446_20180112153206046550_00_004.jpg",
    "574_20180112144145222702_00_004.jpg",
    "424_20180112145115778247_00_004.jpg",
    "446_20180112153211328195_00_004.jpg",
    "574_20180112144220456628_00_004.jpg",
    "424_20180112145118278818_00_004.jpg",
    "446_20180112153212500145_00_004.jpg",
    "574_20180112144220847332_00_004.jpg",
  ]

  for img_file in img_files:
    img_path = os.path.join("/data/deeplearning/mask-rcnn/data/images/", img_file)
    logging.info("Processing image: %s" % img_path)
    vis_image_file = "/data/deeplearning/mask-rcnn/test/" + os.path.basename(img_path)
    instance_bbox, result_img = maskrcnn_user.run(img_path, vis_image_file=vis_image_file)

if __name__ == '__main__':
  main()

