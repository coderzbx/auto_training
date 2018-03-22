# -*- coding:utf-8 -*-

import cv2
import os
import numpy as np
import time

from collections import namedtuple


Label = namedtuple(
    'Label', ['en_name', 'id', 'categoryId', 'color', 'name'])

local_road_chn_labels = {
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
    Label('curb',           20, 20,   (0, 139, 139),      u'虚拟车道线-路缘石'),
    Label('fence',          21, 21,   (255, 106, 106),    u'虚拟车道线-防护栏'),
}

image_dir = "/data/deeplearning/dataset/training/data/local/fisheye"
dir_list = os.listdir(image_dir)
for dir_ in dir_list:
    if not str(dir_).isdigit():
        continue

    start = time.time()
    dir_path = os.path.join(image_dir, dir_)
    list_path = os.path.join(dir_path, dir_+".txt")
    with open(list_path, "r") as f:
        line_str = f.readline()
        while line_str:
            line_str = line_str.strip()
            image_name, label_name = line_str.split(",")

            image_path = os.path.join(dir_path, image_name)
            label_path = os.path.join(dir_path, label_name)
            new_label_path = os.path.join(dir_path, label_name[6:])

            img = cv2.imread(label_path)
            width = img.shape[1]
            height = img.shape[0]

            label_data = np.zeros((height, width), np.uint8)
            label_data[0:height, 0:width] = 255

            for label in local_road_chn_labels:
                color = (label.color[2], label.color[1], label.color[0])
                label_data[np.where((img == color).all(axis=2))] = label.categoryId

            # 校验"其他"类别的占比
            other_count = np.sum(label_data == 255)
            valid_count = width * height * 0.01
            if other_count > valid_count:
                print(image_path)
                os.remove(image_path)
                os.remove(label_path)
            else:
                cv2.imwrite(new_label_path, label_data)

            line_str = f.readline()
    end = time.time()
    print ("process[{}] in {} s".format(dir_, end-start))
