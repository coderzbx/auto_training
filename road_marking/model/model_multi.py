# -*-coding:utf-8-*-

import argparse
import copy
import multiprocessing
import os
import shutil
import time
from multiprocessing import Queue

import cv2
import mxnet as mx
import numpy as np
from PIL import Image

import symbol
from seg_labels import self_server_label
from util import color_normalize
from util import color_scale
from util import interp_preds_as
from util import load_weights
from weighted_loss_layer import WeightedOhemSoftmax
from weighted_loss_layer import WeightedOhemSoftmaxProp


def do_work():
    time1 = time.time()
    model_net = ModelResNetRoad9(gpu_id=0)

    while True:
        if _queue.empty():
            break

        image_path = _queue.get()
        image_name = os.path.basename(image_path)

        name_list = image_name.split(".")
        name_only = name_list[0]

        file_id = name_only + ".png"
        dest_path = os.path.join(dest_dir, file_id)

        if not args.force:
            if os.path.exists(dest_path):
                continue

        try:
            start = time.time()
            with open(image_path, "rb") as f:
                img = f.read()
                model_net.do(image_data=img, dest_file=dest_path)
            end = time.time()
            print("Processed {} in {} ms".format(dest_path, str((end - start) * 1000)))
        except Exception as e:
            print (repr(e))

    time2 = time.time()
    print("finish in {} s".format(time2 - time1))


class ModelResNetRoad9:
    def __init__(self, gpu_id=0):
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        cur_path = os.path.realpath(__file__)
        cur_dir = os.path.dirname(cur_path)
        cur_list = os.listdir(cur_dir)

        model_file_name = "kddata_decoder2_cls12_s2_ep-0065.params"
        for cur_file in cur_list:
            cur_name_list = cur_file.split(".")
            if len(cur_name_list) == 2:
                if cur_name_list[1] == "params":
                    model_file_name = cur_file

        self.weights = os.path.join(cur_dir, model_file_name)

        colours = os.path.join(cur_dir, 'self_clr.png')
        self.label_colours = cv2.imread(colours).astype(np.uint8)

        self.ignore_color = (0, 0, 0)
        clr_dict = {l.name: l.color for l in self_server_label}
        for name, color in clr_dict.items():
            if name == u'Ignore':
                self.ignore_color = color
                break

        network, net_args, net_auxs = load_weights(self.weights)
        self.scale = 1.0/255
        self.mean = [0.2476, 0.2469, 0.250]
        self.std = [0.1147, 0.1056, 0.0966]
        context = [mx.gpu(gpu_id)]
        self.mod = mx.mod.Module(network, context=context)

        self.result_shape = [1024, 2448]
        self.input_shape = [512, 1224]
        # self.batch_data_shape = (1, 3, 1024, 2448)
        self.batch_data_shape = (1, 3, 512, 1224)
        provide_data = [("data", self.batch_data_shape)]
        self.batch_label_shape = (1, 3, 512, 612)
        provide_label = [("softmax_label", self.batch_label_shape)]
        self.mod.bind(provide_data,
                 provide_label,
                 for_training=False,
                 force_rebind=True)
        self.mod.init_params(arg_params=net_args,
                        aux_params=net_auxs)
        self._flipping = False

        self.batch_data = [mx.nd.empty(info[1]) for info in provide_data]
        self.batch_label = [mx.nd.empty(info[1]) for info in provide_label]

        symbol.cfg['workspace'] = 1024
        symbol.cfg['bn_use_global_stats'] = True

    def do(self, image_data, dest_file=None):
        pred_data = None
        try:
            image = np.asarray(bytearray(image_data), dtype="uint8")
            origin_frame = cv2.imdecode(image, cv2.IMREAD_COLOR)

            # read image as rgb
            origin_frame = origin_frame[:, :, ::-1]
            width = origin_frame.shape[1]
            height = origin_frame.shape[0]

            # crop bottom half of the picture
            bottom_half = origin_frame[height-self.result_shape[0]:height, 0:width]

            img = np.array(Image.fromarray(bottom_half.astype(np.uint8, copy=False)).
                           resize((self.input_shape[1], self.input_shape[0]), Image.NEAREST))
            # img = np.array(Image.fromarray(bottom_half.astype(np.uint8, copy=False)))

            img = color_scale(self.scale)(img)
            img = color_normalize(mean=self.mean, std=self.std)(img)
            img = img.transpose(2, 0, 1)
            self.batch_data[0][0] = img

            self.mod.forward(mx.io.DataBatch(data=self.batch_data, label=self.batch_label), is_train=False)

            if self._flipping:
                preds = copy.deepcopy(self.mod.get_outputs())
                flip_batch_data = []
                for batch_split_data in self.batch_data:
                    flip_batch_data.append(mx.nd.array(batch_split_data.asnumpy()[:, :, :, ::-1]))
                self.mod.forward(mx.io.DataBatch(flip_batch_data, label=self.batch_label), is_train=False)
                flip_preds = self.mod.get_outputs()
                merge_preds = []
                for i, pred in enumerate(preds):
                    # change left-lane and right-lane dimension when flipplig
                    flipped_pred = flip_preds[i].asnumpy()[:, :, :, ::-1]
                    flipped_pred[:, [1, 2], :, :] = flipped_pred[:, [2, 1], :, :]
                    merge_preds.append(mx.nd.array((0.5 * pred.asnumpy() + 0.5 * flipped_pred)))
                preds = merge_preds
            else:
                preds = self.mod.get_outputs()

            interp_pred = interp_preds_as(preds[0][0].asnumpy(), self.result_shape, 1)
            pred_label = interp_pred.argmax(0)

            out_pred = np.resize(pred_label, (3, self.result_shape[0], self.result_shape[1]))
            out_pred = out_pred.transpose(1, 2, 0).astype(np.uint8)
            out_rgb = np.zeros(out_pred.shape, dtype=np.uint8)

            cv2.LUT(out_pred, self.label_colours, out_rgb)
            rgb_frame = out_rgb

            # extend result image
            # blank_image = np.zeros((height, width, 3), np.uint8)
            # blank_image[0:height, 0:width] = self.ignore_color
            # blank_image[height-self.result_shape[0]:height, 0:width] = rgb_frame

            blank_image = np.zeros((height, width, 4), np.uint8)
            blank_image[0:self.result_shape[0], 0:width] = (self.ignore_color[0], self.ignore_color[1], self.ignore_color[2], 0)
            blank_image[height-self.result_shape[0]:height, 0:width] = (self.ignore_color[0], self.ignore_color[1], self.ignore_color[2], 255)

            blank_image[height-self.result_shape[0]:height, 0:width, 0] = rgb_frame[:, :, 0]
            blank_image[height-self.result_shape[0]:height, 0:width, 1] = rgb_frame[:, :, 1]
            blank_image[height-self.result_shape[0]:height, 0:width, 2] = rgb_frame[:, :, 2]

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

if __name__ == "__main__":
    _time_start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--force', type=int, required=False, default=0)
    parser.add_argument('--gpu', type=int, default=1, required=False)
    args = parser.parse_args()

    _queue = Queue()
    image_dir = args.dir

    if not os.path.exists(image_dir):
        print("dir[{}] is not exist".format(image_dir))
        exit(0)

    dest_dir = os.path.join(image_dir, "resnet_road")
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    proc_list = []
    file_list = os.listdir(image_dir)
    for id_ in file_list:
        name_list = str(id_).split(".")
        if len(name_list) != 2:
            continue

        name_only = name_list[0]
        name_ext = name_list[1]
        if name_ext != 'png' and name_ext != 'jpg':
            continue
        _queue.put(os.path.join(image_dir, id_))

    all_process = []
    for i in range(args.gpu):
        process = multiprocessing.Process(target=do_work)
        all_process.append(process)

    for process in all_process:
        process.start()
        time.sleep(10)

    for process in all_process:
        process.join()

    _time_end = time.time()
    print("finish in {} s".format(_time_end - _time_start))