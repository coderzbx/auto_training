# -*- coding:utf-8 -*-

import json
import os
import paramiko

from collections import namedtuple

Label = namedtuple(
    'Label', ['en_name', 'id', 'categoryId', 'color', 'name'])

port = 7022
host_ip = "192.168.5.38"
context = "kss-auto-training"
max_packages = 100000

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
    Label('curb',           20, 20,   (0, 139, 139),      u'虚拟车道线-路缘石'),
    Label('fence',          21, 21,   (255, 106, 106),    u'虚拟车道线-防护栏'),
}


def create_ssh_client(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client


def get_file(file_dir, file_list, src_len):
    files = os.listdir(file_dir)

    for _file in files:
        _dir = os.path.join(file_dir, _file)
        if os.path.isdir(_dir):
            get_file(_dir, file_list, src_len)
        else:
            _name_list = str(_file).split(".")
            if len(_name_list) < 2:
                continue
            # _file_name = _name_list[len(_name_list) - 2]
            _file_ext = _name_list[len(_name_list) - 1]

            if _file_ext not in ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]:
                continue

            _file_path = _dir[src_len+1:]

            if isinstance(file_list, list):
                file_list.append(_file_path)
    pass


class Task:
    def __init__(self, package_index, src_path, dest_path, dest_label, exit_flag=False):
        self.package_index = package_index
        self.src_path = src_path
        self.dest_path = dest_path
        self.dest_label = dest_label
        self.exit_flag = exit_flag


class RemoteTask:
    def __init__(self, package_index, src_path, dest_path, exit_flag=False):
        self.package_index = package_index
        self.src_path = src_path
        self.dest_path = dest_path
        self.exit_flag = exit_flag


class OnlineTask:
    def __init__(self, track_point_id, task_id, image_data, label_data, exit_flag=False):
        self.track_point_id = track_point_id
        self.task_id = task_id
        self.image_data = image_data
        self.label_data = label_data
        self.exit_flag = exit_flag


class ServerResponse:
    def __init__(self, err_code, err_info=None, result=None):
        self.err_code = int(err_code)
        self.result = result

        self.err_info = ""
        if err_info is None:
            if self.err_code == 0:
                self.err_info = "success"
            elif self.err_code == 1:
                self.err_info = "images do not exist"
            elif self.err_code == 2:
                self.err_info = "labels do not exist"
            elif self.err_code == 3:
                self.err_info = "training dataset do not exist"
            elif self.err_code == 99:
                self.err_info = "unknown error"

    def generate_response(self):
        # {“code”:”0”, ”msg”:”success”}
        json_data = {
            "code": str(self.err_code),
            "msg": self.err_info
        }
        if self.result is not None:
            json_data["result"] = self.result

        json_str = json.dumps(json_data)
        return json_str