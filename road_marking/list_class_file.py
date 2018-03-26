# -*-coding:utf-8-*-

import os
import time
import cv2
import numpy as np
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


if __name__ == '__main__':
    package_dir = '/data/deeplearning/dataset/training/data/released_temp'
    target_id = 5
    target_list = {}

    dir_list = os.listdir(package_dir)
    for dir_name in dir_list:
        if dir_name.isdigit():
            continue
        dir_path = os.path.join(package_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue
        packages = os.listdir(dir_path)
        packages.sort()
        for package in packages:
            start_ = time.time()
            if not str(package).isdigit():
                continue

            package_path = os.path.join(dir_path, package)
            file_list = os.listdir(package_path)
            for file_id in file_list:
                if not file_id.endswith("png") or file_id.startswith("label") or file_id.startswith("ext"):
                    continue

                label_data = cv2.imread(os.path.join(package_path, file_id), cv2.IMREAD_GRAYSCALE)
                for label in local_road_chn_labels:
                    sum_id = label.categoryId
                    target_count = np.sum(label_data == sum_id)

                    if target_count > 0:
                        if sum_id not in target_list:
                            target_list[sum_id] = []
                        target_list[sum_id].append(os.path.join(package_path, file_id))
            end_ = time.time()
            print("finish filter {} in {} s".format(package_path, (end_ - start_)))

    for class_id, path_list in target_list.items():
        with open("./class_list_{}.txt".format(class_id), "w") as f:
            for path_id in path_list:
                f.write("{}\n".format(path_id))
