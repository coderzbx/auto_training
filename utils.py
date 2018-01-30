# -*- coding:utf-8 -*-

import json
import os
import paramiko

from collections import namedtuple

Label = namedtuple(
    'Label', ['name', 'id', 'classId', 'className', 'categoryId',
              'hasInstances', 'ignoreInEval', 'color'])

Label1 = namedtuple(
    'Label', ['id', 'categoryId', 'color', 'name'])

port = 7022
host_ip = "192.168.5.38"
context = "kss-auto-training"
max_packages = 100000

self_road_chn_labels = {
    Label(u'车道标线', 0, 0, 'Lane', 0, True, False, (255, 0, 0)),
    Label(u'左侧道路边缘线', 1, 1, 'Symbol', 1, True, False, (255, 192, 203)),
    Label(u'右侧道路边缘线', 2, 2, 'SpecialLane', 2, True, False, (139, 0, 139)),
    Label(u'纵向减速标线', 3, 1, 'Symbol', 3, True, False, (32, 128, 192)),
    Label(u'专用车道标线', 4, 2, 'SpecialLane', 4, True, False, (192, 128, 255)),
    Label(u'停止线', 5, 3, 'StopLane', 5, True, False, (255, 128, 64)),
    Label(u'减速让行标线', 6, 4, 'SlowLane', 6, True, False, (0, 255, 255)),
    Label(u'减速标线/减速带', 7, 5, 'SlowNet', 7, True, False, (128, 128, 255)),
    Label(u'人行横道', 8, 14, 'SlowNet', 8, True, False, (128, 192, 192)),
    Label(u'路面连接带', 9, 15, 'SlowNet', 9, True, False, (128, 128, 192)),
    Label(u'出入口标线', 10, 6, 'InOut', 10, True, False, (128, 128, 0)),
    Label(u'文字符号类', 11, 7, 'FishLane', 11, True, False, (0, 0, 255)),
    Label(u'导流线（鱼刺线）', 12, 8, 'FishLane', 12, True, False, (0, 255, 0)),
    Label(u'停止网格标线', 13, 9, 'StopNet', 13, True, False, (255, 255, 0)),
    Label(u'车距确认线', 14, 10, 'Sidewalk', 14, True, False, (255, 128, 255)),
    Label(u'道路', 15, 11, 'Road', 15, True, False, (192, 192, 192)),
    Label(u'其他', 16, 12, 'Other', 16, True, False, (64, 64, 32)),
    Label(u'车辆及路面上其他物体', 17, 12, 'Other', 17, True, False, (128, 0, 0)),
    Label(u'Ignore', 18, 13, 'Other', 18, True, False, (0, 0, 0)),
}

self_road_simple_labels = {
    Label1(0, 0,     (255, 0, 0),        u'车道标线'),
    Label1(1, 1,     (255, 192, 203),    u'左侧道路边缘线'),
    Label1(2, 2,     (139, 0, 139),      u'右侧道路边缘线'),
    Label1(3, 3,     (32, 128, 192),     u'纵向减速标线'),
    Label1(4, 4,     (192, 128, 255),    u'专用车道标线'),
    Label1(5, 5,     (255, 128, 64),     u'停止线'),
    Label1(6, 6,     (0, 255, 255),      u'减速让行标线'),
    Label1(7, 7,     (128, 128, 255),    u'减速标线/减速带'),
    Label1(8, 8,     (128, 192, 192),    u'人行横道'),
    Label1(9, 9,     (128, 128, 192),    u'路面连接带'),
    Label1(10, 10,   (128, 128, 0),      u'出入口标线'),
    Label1(11, 11,   (0, 0, 255),        u'文字符号类'),
    Label1(12, 12,   (0, 255, 0),        u'导流线（鱼刺线）'),
    Label1(13, 13,   (255, 255, 0),      u'停止网格标线'),
    Label1(14, 14,   (255, 128, 255),    u'车距确认线'),
    Label1(15, 15,   (192, 192, 192),    u'道路'),
    Label1(16, 16,   (64, 64, 32),       u'其他'),
    Label1(17, 17,   (128, 0, 0),        u'车辆及路面上其他物体'),
    Label1(18, 18,   (0, 0, 0),          u'Ignore'),
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