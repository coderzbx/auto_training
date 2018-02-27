
# -*- coding:utf-8 -*-

import tornado.web
import os
import json
import time
import multiprocessing
import logging
import shutil
import argparse

import cv2
import scp
from stat import S_ISDIR
from collections import namedtuple

import numpy as np

max_packages = 10000

Label = namedtuple(
    'Label', ['en_name', 'id', 'categoryId', 'color', 'name'])

self_road_chn_labels = {
    Label('other',          0, 0,     (64, 64, 32),       u'其他'),
    Label('ignore',         1, 1,     (0, 0, 0),          u'Ignore'),
    Label('lane',           2, 2,     (255, 0, 0),        u'车道标线'),
    Label('left',           3, 3,     (255, 192, 203),    u'左侧道路边缘线'),
    Label('right',          4, 4,     (139, 0, 139),      u'右侧道路边缘线'),
    Label('v_slow',         5, 5,     (32, 128, 192),     u'纵向减速标线'),
    Label('bus_lane',       6, 6,     (192, 128, 255),    u'专用车道标线'),
    Label('stop',           7, 7,     (255, 128, 64),     u'停止线'),
    Label('slow_let',       8, 8,     (0, 255, 255),      u'减速让行标线'),
    Label('slow_zone',      9, 9,     (128, 128, 255),    u'减速标线/减速带'),
    Label('sidewalk',       10, 10,   (128, 192, 192),    u'人行横道'),
    Label('connection',     11, 11,   (128, 128, 192),    u'路面连接带'),
    Label('stop_station',   12, 12,   (240, 128, 128),    u'停靠站标线'),
    Label('in_out',         13, 13,   (128, 128, 0),      u'出入口标线'),
    Label('symbol',         14, 14,   (0, 0, 255),        u'文字符号类'),
    Label('fish_lane',      15, 15,   (0, 255, 0),        u'导流线（鱼刺线）'),
    Label('stop_gird',      16, 16,   (255, 255, 0),      u'停止网格标线'),
    Label('distance',       17, 17,   (255, 128, 255),    u'车距确认线'),
    Label('road',           18, 18,   (192, 192, 192),    u'道路'),
    Label('objects',        19, 19,   (128, 0, 0),        u'车辆及路面上其他物体'),
    Label('emergency',      20, 20,   (229, 152, 102),    u'虚拟应急车道线'),
}


class Task:
    def __init__(self, package_index, src_path, dest_path, dest_label, exit_flag=False):
        self.package_index = package_index
        self.src_path = src_path
        self.dest_path = dest_path
        self.dest_label = dest_label
        self.exit_flag = exit_flag

task_queue = multiprocessing.Manager().Queue()
_queue = multiprocessing.Manager().Queue()


class ProcessLabelHandler():
    def prepare_crop(self, _csv_file):
        root_dir = os.path.dirname(_csv_file)
        csv_name = os.path.basename(_csv_file)

        name_list = csv_name.split(".")
        csv_name = name_list[0]
        if csv_name.startswith("ext-"):
            csv_name = csv_name[4:]
        package_index = csv_name

        list_file = csv_name + ".csv"
        list_path = os.path.join(root_dir, list_file)

        temp_dir = os.path.join(dest_dir, package_index)
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)

        dest_list_path = os.path.join(temp_dir, list_file)
        if os.path.exists(list_path):
            shutil.copy(list_path, dest_list_path)
        else:
            list_file = csv_name + ".txt"
            list_path = os.path.join(root_dir, list_file)
            dest_list_path = os.path.join(temp_dir, list_file)
            if os.path.exists(list_path):
                shutil.copy(list_path, dest_list_path)

        with open(_csv_file, "r") as f:
            line_str = f.readline()

            while line_str:
                image, label = line_str.split(",")
                image = image.strip()
                label = label.strip()

                if not image.startswith("ext-") or not label.startswith("ext-"):
                    line_str = f.readline()
                    continue

                origin_image = os.path.join(root_dir, image[4:])
                if not os.path.exists(origin_image):
                    line_str = f.readline()
                    print ("package:{}, file missing:[{}]=>[{}]".format(package_index, image, label))
                    continue
                else:
                    dest_origin_image = os.path.join(temp_dir, image[4:])
                    shutil.copy(origin_image, dest_origin_image)

                if not os.path.exists(os.path.join(root_dir, label)):
                    line_str = f.readline()
                    print ("package:{}, file missing:[{}]=>[{}]".format(package_index, image, label))
                    continue

                _src = os.path.join(root_dir, label)
                _dest_id2 = label[4:]
                _dest_label = os.path.join(temp_dir, _dest_id2)
                _dest_label = _dest_label.strip()

                task = Task(package_index, _src, _dest_label, None)
                _queue.put(task)

                line_str = f.readline()

    def do_work(self):
        if _queue.empty():
            return

        while not _queue.empty():
            task = _queue.get()

            if not isinstance(task, Task):
                break

            if task.exit_flag:
                break

            time1 = time.time()

            _src = task.src_path
            _dest = task.dest_path

            img = cv2.imread(_src)
            width = img.shape[1]
            height = img.shape[0]

            self.pixel = int((width - 2448) / 2)
            crop_img = img[self.pixel:height - self.pixel, self.pixel:width - self.pixel]
            cv2.imwrite(_dest, crop_img)

            time2 = time.time()
            print ("process[{}/{}] in {} s".format(task.package_index, _src, time2 - time1))

    def transform(self):
        while not task_queue.empty():
            task = task_queue.get()

            if not isinstance(task, Task):
                break

            if task.exit_flag:
                break

            image_path = task.src_path
            result_path = task.dest_path

            time1 = time.time()

            img = cv2.imread(image_path)
            if img is None:
                print ("image is none[{}/{}]".format(task.package_index, image_path))
                continue

            width = img.shape[1]
            height = img.shape[0]

            # other_category = -1
            label_data = np.zeros((height, width), np.uint8)
            label_data[0:height, 0:width] = 255
            # for label in self_road_chn_labels:
            #     if label.name == u"其他":
            #         other_category = label.categoryId
            #         break

            for label in self_road_chn_labels:
                color = (label.color[2], label.color[1], label.color[0])
                label_data[np.where((img == color).all(axis=2))] = label.categoryId

            # 校验"其他"类别的占比
            other_count = np.sum(label_data == 255)
            valid_count = width*height*0.01
            if other_count > valid_count:
                label_name = os.path.basename(image_path)
                file_name = label_name.split(".")[0]
                if file_name.startswith("ext-"):
                    file_name = file_name[4:]
                if file_name.startswith("label-"):
                    file_name = file_name[6:]
                origin_image_path = os.path.join(os.path.dirname(image_path), file_name+".jpg")
                print ("label[{}/{}] not qualified".format(task.package_index, image_path))

                if os.path.exists(origin_image_path):
                    os.remove(origin_image_path)
                if os.path.exists(image_path):
                    os.remove(image_path)
            else:
                cv2.imwrite(result_path, label_data)

            time2 = time.time()
            print ("process[{}/{}] in {} s".format(task.package_index, image_path, time2 - time1))

    def clean_dir(self, _csv_file):
        image_list = []
        package_index = ""
        root_dir = os.path.dirname(_csv_file)
        csv_name = os.path.basename(_csv_file)
        image_list.append(str(csv_name).strip())

        if csv_name.startswith("ext-"):
            origin_csv = csv_name[4:]
            image_list.append(str(origin_csv).strip())

            name_list = origin_csv.split(".")
            new_file_name = name_list[0] + ".txt"
            package_index = name_list[0]
            image_list.append(new_file_name)

        with open(_csv_file, "r") as f:
            line_str = f.readline()

            while line_str:
                image, label = line_str.split(",")
                image = str(image).strip()
                label = str(label).strip()

                image_path = os.path.join(root_dir, image)
                label_path = os.path.join(root_dir, label)

                if os.path.exists(image_path) and os.path.exists(label_path):
                    image_list.append(image)
                    image_list.append(label)

                    if image.startswith("ext-"):
                        origin_image = image[4:]
                        image_list.append(str(origin_image).strip())

                        name_list = origin_image.split(".")
                        if len(name_list) == 2:
                            annotation_file = name_list[0] + ".png"
                            image_list.append(str(annotation_file).strip())
                    if label.startswith("ext-"):
                        origin_label = label[4:]
                        image_list.append(str(origin_label).strip())

                if not os.path.exists(image_path):
                    if os.path.exists(label_path):
                        print ("package:{}, not match:[{}] not exist=>[{}] exist".format(package_index, image, label))
                        if os.path.exists(label_path):
                            os.remove(label_path)
                    else:
                        print ("package:{}, file missing:[{}]=>[{}]".format(package_index, image, label))
                else:
                    if not os.path.exists(label_path):
                        print ("package:{}, not match:[{}] exist=>[{}] not exist".format(package_index, image, label))
                        if os.path.exists(image_path):
                            os.remove(image_path)

                line_str = f.readline()

        file_list = os.listdir(root_dir)
        for file_id in file_list:
            file_id = str(file_id).strip()
            if file_id not in image_list:
                file_delete = os.path.join(root_dir, file_id)
                print ("package:{}, file wrong:[{}]".format(package_index, file_id))
                if os.path.exists(file_delete):
                    if os.path.isfile(file_delete):
                        os.remove(file_delete)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)
    parser.add_argument('--start', type=int, required=True, default=1)
    parser.add_argument('--step', type=int, required=False, default=20)
    args = parser.parse_args()

    time1 = time.time()

    _proc_handler = ProcessLabelHandler()
    src_dir = args.src_dir
    dest_dir = args.dest_dir
    start = args.start

    # {
    #     “images”:”原始图片目录，缺省默认为当前目录的images目录”,
    #     “pixel”:外边框像素大小，缺省默认为50,
    # }

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    err_code = 0
    for dir in range(max_packages):
        csv_name = "ext-" + str(dir) + ".csv"
        csv_file = os.path.join(src_dir, str(dir), csv_name)
        if not os.path.exists(csv_file):
            continue

        _proc_handler.clean_dir(_csv_file=csv_file)
        _proc_handler.prepare_crop(_csv_file=csv_file)

    file_list = list()
    pixel = 50

    task_count = _queue.qsize()
    # start to process
    task = Task(None, None, None, None, True)
    _queue.put(task)

    process = multiprocessing.Process(target=_proc_handler.do_work)
    process.start()
    process.join()

    for i in range(max_packages):
        _sub_dir = os.path.join(dest_dir, str(i))
        if not os.path.exists(_sub_dir):
            continue

        _dest_dir = os.path.join(dest_dir, str(i))
        if not os.path.exists(_dest_dir):
            os.mkdir(_dest_dir)

        origin_list = os.listdir(_sub_dir)

        for _image in origin_list:
            _image = _image.strip()
            if not _image.startswith("label-"):
                continue

            name_list = _image.split('.')
            if len(name_list) < 2:
                continue

            ext_name = name_list[1]
            if ext_name != 'png' and ext_name != 'jpg':
                continue

            # start with label-
            label_file = name_list[0]
            if label_file.startswith("label-"):
                label_file = label_file[6:]
            anna_file = label_file + ".png"

            origin_name = label_file + ".jpg"
            image_path = os.path.join(_sub_dir, _image)
            origin_image = os.path.join(_sub_dir, origin_name)
            if not os.path.exists(origin_image):
                os.remove(image_path)
                continue

            result_path = os.path.join(dest_dir, str(i), anna_file)

            task = Task(str(i), image_path, result_path, None, False)
            task_queue.put(task)
    for i in range(20):
        task = Task(None, None, None, None, True)
        task_queue.put(task)

    all_processes = []
    for i in range(20):
        process = multiprocessing.Process(target=_proc_handler.transform)
        all_processes.append(process)

    for process in all_processes:
        process.start()

    for process in all_processes:
        process.join()

    time2 = time.time()
    result_obj = {
        "count": str(task_count),
        "time": str(time2 - time1) + " s"
    }
    print(result_obj)
