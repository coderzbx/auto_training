#!/usr/bin/env python
#encoding=utf-8

import sys
import os
import json
import time
import copy
import logging
import mxnet as mx
import numpy as np
import shutil
from PIL import Image

from util import KdSeg12Id, KdSeg12Id2Color
from seg_labels import self_server_label
from util import color_normalize
from util import color_scale
from util import interp_preds_as
from util import load_weights

from weighted_loss_layer import WeightedOhemSoftmax
from weighted_loss_layer import WeightedOhemSoftmaxProp
import metric
import symbol

symbol.cfg['workspace'] = 1024
symbol.cfg['bn_use_global_stats'] = True


class CheckData(object):
    def __init__(self, gpu_id=0):

        self.scale = 1.0 / 255
        self.mean = [0.2476, 0.2469, 0.250]
        self.std = [0.1147, 0.1056, 0.0966]

        cur_path = os.path.realpath(__file__)
        cur_dir = os.path.dirname(cur_path)
        cur_list = os.listdir(cur_dir)

        model_file_name = "kddata_decoder2_cls12_s2_ep-0140.params"
        self.weights_file = os.path.join(cur_dir, model_file_name)

        self.temp_dir = "/data/deeplearning/dataset/training/data/test"
        self.data_dir = "/data/deeplearning/dataset/training/data/check"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self.use_half_image = True
        self.save_result_or_not = True

        # Thresholds
        self.iou_thresh_low = 0.2
        self.min_pixel_num = 2000

        # device
        context = [mx.gpu(gpu_id)]
        network, net_args, net_auxs = self.load_weights(self.weights_file)

        # Set batch size
        self.batch_size = 1

        # Module
        batch_data_shape = (self.batch_size, 3, 512, 1224)
        # batch_data_shape = (self.batch_size, 3, 1024, 2448)
        provide_data = [("data", batch_data_shape)]
        batch_label_shape = (self.batch_size, 512, 1224)
        # batch_label_shape = (self.batch_size, 1024, 2448)
        provide_label = [("softmax_label", batch_label_shape)]
        self.mod = mx.mod.Module(network, context=context)
        self.mod.bind(provide_data,
                      provide_label,
                      for_training=False,
                      force_rebind=True)
        self.mod.init_params(arg_params=net_args, aux_params=net_auxs)

        # Upsampling module
        input_data = mx.symbol.Variable(name='data')
        upsampling = mx.symbol.UpSampling(input_data,
                                          scale=2,
                                          num_filter=12,
                                          sample_type='bilinear',
                                          name="upsampling12",
                                          workspace=512)
        argmax_sym = mx.sym.argmax(data=upsampling, axis=1)
        self.upsample_mod = mx.mod.Module(argmax_sym, context=context,
                            data_names = ['data'],
                            label_names = [])
        self.upsample_mod.bind(data_shapes=[('data', (self.batch_size, 12L, 256L, 612L))],
                 label_shapes=None,
                 for_training=False,
                 force_rebind=True)
        initializer = mx.init.Bilinear()
        self.upsample_mod.init_params(initializer=initializer)

        # batch data & batch label
        self.batch_data = [mx.nd.empty(info[1]) for info in provide_data]
        self.batch_label = [mx.nd.empty(info[1]) for info in provide_label]

        # metric
        self.seg_metric = metric.SegMetric()
        self.seg_metric.set_nclass(12)

    def __del__(self):
        pass

    def load_weights(self, weights_file):
        assert os.path.exists(weights_file)
        prefix = weights_file.split("_ep-")[0] + "_ep"
        epoch = int(weights_file.split("_ep-")[1].split(".")[0])
        logging.info("prefix: {}, epoch: {}".format(prefix, epoch))
        network, net_args, net_auxs = mx.model.load_checkpoint(prefix, epoch)
        return network, net_args, net_auxs

    def read_image(self, img_file):
        img = Image.open(img_file)
        if self.use_half_image:
            w, h = img.size
            w = w // 2
            h = h // 2
            img = img.resize((w, h))
        img = np.array(img)

        # preprocess data for img data: substract mean, divided by std
        img = color_scale(self.scale)(img)
        img = color_normalize(mean=self.mean, std=self.std)(img)
        img = img.transpose(2, 0, 1)
        return img

    def read_label(self, gt_label_file):
        img = Image.open(gt_label_file)
        if self.use_half_image:
            w, h = img.size
            w = w // 2
            h = h // 2
            img = img.resize((w, h), Image.NEAREST)
        return np.array(KdSeg12Id()(img))

    def feed_batch_data(self, image_file_list, cursor, cursor_to):
        for i in range(cursor_to - cursor):
            image_file = image_file_list[cursor + i]
            # img_file = os.path.join(self.data_dir, image_file)
            img_file = image_file
            logging.info("Processing file: %s" % (img_file))
            self.batch_data[0][i] = self.read_image(img_file)

    def feed_batch_label(self, image_file_list, cursor, cursor_to):
        for i in range(cursor_to - cursor):
            image_file = image_file_list[cursor + i]
            gt_label_file = image_file[:-3] + "png"
            # gt_label_file = os.path.join(self.data_dir, gt_label_file)
            assert os.path.exists(gt_label_file), "File not exists: {}".format(gt_label_file)
            label_data = self.read_label(gt_label_file)
            self.batch_label[0][i] = label_data
            if os.path.basename(gt_label_file) == '668_20180113163956271507_00_004.png':
                print("got you")
                car1_count = np.sum(label_data == 10)
                print(car1_count)

    def gpu_upsampling(self, data):
        self.upsample_mod.forward(mx.io.DataBatch(data=data), is_train=False)
        outputs = self.upsample_mod.get_outputs()
        return outputs

    def predict(self):
        self.mod.forward(mx.io.DataBatch(
                    data=self.batch_data, label=self.batch_label), is_train=False)
        preds = self.mod.get_outputs()
        pred_label = self.gpu_upsampling(preds)
        return pred_label

    def save_result(self, image_file_list, cursor, cursor_to, pred_label):
        pred_label = pred_label.asnumpy().astype(np.int)
        for i in range(cursor_to - cursor):
            save_img = Image.fromarray(KdSeg12Id2Color()(pred_label[i]))
            image_file = image_file_list[cursor + i]
            # label_file = os.path.join(self.predict_dir, image_file)
            label_file = os.path.join(self.predict_dir, os.path.basename(image_file)[:-3]+"png")
            save_img.save(label_file)

    def eval_predict_gt(self, pred, gt_label):
        assert self.batch_size == 1, "Single batch only."
        assert len(pred) == len(gt_label)
        self.seg_metric.reset()
        self.seg_metric.update(gt_label, pred)
        names, values = self.seg_metric.get()
        for name, value in zip(names, values):
            if "accs" == name:
                accs = value
            if "ious" == name:
                ious = value
        # iou info: (cls_id, iou, pred_cls_pixel_num, gt_label_cls_pixle_num)
        iou_info_array = []
        for idx, iou in enumerate(ious):
            if np.isnan(iou):
                continue
            pred_cls_pixel_num = np.sum(pred[0].asnumpy().astype(np.int) == idx)
            gt_label_cls_pixle_num = np.sum(gt_label[0].asnumpy().astype(np.int) == idx)
            iou_info_array.append((idx, iou, pred_cls_pixel_num, gt_label_cls_pixle_num))
        return self.check_false(iou_info_array)

    def check_false(self, iou_info_array):
        # sort based on iou
        iou_info_array = sorted(iou_info_array, key=lambda d: d[1])
        logging.info(iou_info_array)
        for iou_info in iou_info_array:
            max_pixel_num = max(iou_info[2:])
            if iou_info[1] < self.iou_thresh_low and max_pixel_num >= self.min_pixel_num:
                return iou_info
        return None

    def run(self):
        check_result = {}
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self.predict_dir = os.path.join(self.data_dir, "predict")
        if not os.path.exists(self.predict_dir):
            os.makedirs(self.predict_dir)

        self.doubt_dir = os.path.join(self.data_dir, "doubt")
        if not os.path.exists(self.doubt_dir):
            os.makedirs(self.doubt_dir)

        doubt_file = os.path.join(self.data_dir, "doubt.list")
        out_f = open(doubt_file, "w")

        image_file_list = []
        package_dir_list = os.listdir(self.temp_dir)
        for package_id in package_dir_list:
            package_dir = os.path.join(self.temp_dir, package_id)
            for image_file in os.listdir(package_dir):
                if image_file.endswith("csv") or image_file.endswith("txt") \
                        or image_file.startswith("label") or image_file.startswith("ext")\
                        or image_file.endswith("png"):
                    continue
                image_file_list.append(os.path.join(package_dir, image_file))
        assert len(image_file_list) != 0
        logging.info("number_of_images: %d" % (len(image_file_list)))

        cursor = 0
        number_of_samplers = len(image_file_list)
        while cursor < number_of_samplers:
            cursor_to = min(number_of_samplers, cursor + self.batch_size)
            try:
                self.feed_batch_data(image_file_list, cursor, cursor_to)
            except Exception as e:
                print(repr(e))

            # Predicting
            tic = time.time()
            pred_label = self.predict()

            # Save result
            if self.save_result_or_not:
                self.save_result(image_file_list, cursor, cursor_to, pred_label[0])
            logging.info("Speed: %.2f images/sec" % (self.batch_size * 1.0 / (time.time() - tic)))

            # Evalute it: Only the first image, the batchsize should be set to 1
            self.feed_batch_label(image_file_list, cursor, cursor_to)
            ret = self.eval_predict_gt(pred_label, self.batch_label)
            if ret is not None:
                class_index, min_iou, pred_cls_pixel_num, gt_label_cls_pixle_num = ret
                # image_file = os.path.join(self.data_dir, image_file_list[cursor])
                image_file = image_file_list[cursor]
                # label_file = os.path.join(self.data_dir, "label-" + image_file_list[cursor][:-3] + "png")
                label_file = image_file_list[cursor][:-3] + "png"
                logging.info("{}\t{}\t{}".format(image_file, class_index, min_iou))

                # File list
                out_f.write("{}\t{}\t{}\t{}\t{}".format(image_file, class_index, min_iou,
                            pred_cls_pixel_num, gt_label_cls_pixle_num))

                class_name = ""
                for label in self_server_label:
                    if label.categoryId == int(class_index):
                        class_name = label.name
                        break
                package = os.path.basename(os.path.dirname(image_file))
                if not package in check_result:
                    check_result[package] = []

                single_info = {
                    "file": os.path.basename(image_file),
                    "class_name": class_name,
                    "class_id": class_index,
                    "min_iou": min_iou,
                    "pred_cls": pred_cls_pixel_num,
                    "gt_cls": gt_label_cls_pixle_num
                }
                check_result[package].append(single_info)
                # check_result[package].append("{}\t{}\t{}\t{}\t{}".format(image_file, class_index, min_iou,
                #             pred_cls_pixel_num, gt_label_cls_pixle_num))
                out_f.write("\n")
                # Copy file
                assert os.path.exists(image_file)
                assert os.path.exists(label_file)
                shutil.copy(image_file, self.doubt_dir)
                shutil.copy(label_file, self.doubt_dir)

            # Update cursor
            cursor += self.batch_size

        out_f.close()
        return check_result

if __name__ == '__main__':
    log_fmt = '%(levelname)s: %(asctime)s: %(filename)s:%(lineno)d * %(thread)d %(message)s'
    logging.basicConfig(format=log_fmt,
                        datefmt='%Y%m%d-%H:%M:%S',
                        level=logging.INFO)
    params = {
        "use_half_image" : True,
        "save_result_or_not" : True,
        "iou_thresh_low" : 0.5,
        "min_pixel_num" : 2000
    }
    check_data = CheckData(gpu_id=0)
    check_ret = check_data.run()
    print(check_ret)


