# -*- coding:utf-8 -*-

import tornado.web
import os
import json
import time
import multiprocessing
import logging

import cv2
import scp
import numpy as np

from utils import ServerResponse
from utils import get_file
from utils import RemoteTask
from utils import create_ssh_client
from utils import host_ip, max_packages

import global_queue


class TaskDivideRemoteHandler(tornado.web.RequestHandler):
    def initialize(self):
        self.file_list = list()
        self.pixel = 50
        self.step = 20
        self.start = 1
        self.callback = False

        self.logger = logging.getLogger("auto-training")

        self.src_dir = "/data/deeplearning/dataset/training/data/images"
        self.dest_dir = "/data/deeplearning/dataset/training/data/packages"

        self.src_scp_ip = "192.168.5.38"
        self.src_scp_port = 22
        self.src_scp_user = "kddev"
        self.src_scp_passwd = "12345678"

        self.src_ssh = None
        self.src_scp = None
        self.src_sftp = None

        if self.src_scp_ip != host_ip:
            self.src_ssh = create_ssh_client(
                server=self.src_scp_ip,
                port=self.src_scp_port,
                user=self.src_scp_user,
                password=self.src_scp_passwd
            )
            self.src_scp = scp.SCPClient(self.src_ssh.get_transport())
            self.src_sftp = self.src_ssh.open_sftp()

        self.dest_scp_ip = "192.168.5.38"
        self.dest_scp_port = 22
        self.dest_scp_user = "kddev"
        self.dest_scp_passwd = "12345678"

        self.dest_ssh = None
        self.dest_scp = None
        self.dest_sftp = None
        if self.dest_scp_ip != host_ip:
            self.dest_ssh = create_ssh_client(
                server=self.dest_scp_ip,
                port=self.dest_scp_port,
                user=self.dest_scp_user,
                password=self.dest_scp_passwd
            )
            self.dest_scp = scp.SCPClient(self.dest_ssh.get_transport())
            self.dest_sftp = self.dest_ssh.open_sftp()

    def __delete__(self, instance):
        if self.src_scp:
            self.src_scp.close()
        if self.src_sftp:
            self.src_sftp.close()
        if self.src_ssh:
            self.src_ssh.close()
        if self.dest_scp:
            self.dest_scp.close()
        if self.dest_sftp:
            self.dest_sftp.close()
        if self.dest_ssh:
            self.dest_ssh.close()

    def get(self, *args, **kwargs):
        time_start = time.time()

        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Content-type', 'application/json')

        # {
        #     “images”:”原始图片目录，缺省默认为当前目录的images目录”,
        #     “step”:单个任务包的数量，缺省默认为20,
        #     “pixel”:外边框像素大小，缺省默认为50,
        #     “start”:任务包编号开始序号，缺省默认自动记录,
        # }

        try:

            step = self.get_argument("step", "20")
            step = int(step)
            self.step = step

            if self.src_scp_ip != host_ip:
                # 拷贝文件
                if not os.path.exists(self.src_dir):
                    os.makedirs(self.src_dir)
                src_list = self.src_sftp.listdir(self.src_dir)
                for src_file in src_list:
                    src_path = os.path.join(self.src_dir, src_file)
                    file_name = os.path.basename(src_path)
                    name_list = file_name.split(".")
                    if len(name_list) == 2 and name_list[1] in ["jpg", "png", "jpeg", "JPG", "PNG", "JPEG"]:
                        self.src_scp.get(src_path, src_path)

            if not os.path.exists(self.src_dir):
                task_count = 0
                err_code = 1
            else:
                err_code = 0
                dir_list = os.listdir(self.dest_dir)
                cur_max = 0
                for _dir in dir_list:
                    if _dir.isdigit():
                        if int(_dir) > cur_max:
                            cur_max = int(_dir)

                if self.start <= cur_max:
                    self.start = cur_max + 1

                self.prepare_task(
                    src_dir=self.src_dir,
                    dest_dir=self.dest_dir,
                    start_package=self.start,
                    cnt_per_package=self.step
                )

                task_count = global_queue.remote_extend_queue.qsize()

                _task = RemoteTask(
                    package_index=None,
                    src_path=None,
                    dest_path=None,
                    exit_flag=True
                )
                global_queue.remote_extend_queue.put(_task)

                process = multiprocessing.Process(target=self.do)
                process.daemon = True
                process.start()
                process.join()

            time_end = time.time()
            result_obj = {
                "count": str(task_count),
                "time": str(time_end - time_start) + " s"
            }
            resp = ServerResponse(err_code=err_code, err_info=None, result=result_obj)
            resp_str = resp.generate_response()
            self.logger.info(resp_str)

            self.write(resp_str)
        except Exception as err:
            err_info = err.args[0]
            json_res = {"code": "99", "msg": str(err_info)}
            self.write(json.dumps(json_res))
            self.logger.error(json.dumps(json_res))
        except:
            self.write('{"code": "99", "msg": "unknown exception"}')
            self.logger.error('{"code": "99", "msg": "unknown exception"}')

        self.finish()

    # start: 开始包的索引
    # step: 每个包内文件个数
    def prepare_task(self, src_dir, dest_dir, start_package, cnt_per_package):
        # 生成标注任务包
        src_len = len(src_dir)
        get_file(src_dir, self.file_list, src_len)

        self.file_list.sort()
        total_count = len(self.file_list)
        file_index = 0
        total_index = 0
        package_index = start_package
        package_list = {}
        for _file_path in self.file_list:
            total_index += 1

            _file_id = os.path.basename(_file_path)

            package_dir = os.path.join(dest_dir, str(package_index))
            if not os.path.exists(package_dir):
                os.makedirs(package_dir)

            image_file = _file_id

            _file_name = _file_id.split(".")
            _file_name = _file_name[0]
            label_file = "label-" + _file_name + ".png"
            package_list[image_file] = label_file

            src_path = os.path.join(src_dir, _file_path)

            _image = cv2.imread(src_path)
            if _image is None:
                print(src_path)
                continue

            ext_image_name = "ext-"+_file_id
            dest_path = os.path.join(package_dir, ext_image_name)

            _task = RemoteTask(
                package_index=str(package_index),
                src_path=src_path,
                dest_path=dest_path
            )
            global_queue.remote_extend_queue.put(_task)

            file_index += 1
            if file_index == cnt_per_package:
                dest_file = "ext-" + str(package_index) + ".csv"
                dest_file_path = os.path.join(dest_dir, str(package_index), dest_file)

                with open(dest_file_path, "w") as f:
                    for _image, _label in package_list.items():
                        _str = "ext-{},ext-{}\n".format(_image, _label)
                        f.write(_str)

                package_list = {}
                file_index = 0
                package_index += 1
            elif total_index == total_count:
                dest_file = "ext-" + str(package_index) + ".csv"
                dest_file_path = os.path.join(dest_dir, str(package_index), dest_file)

                with open(dest_file_path, "w") as f:
                    for _image, _label in package_list.items():
                        _str = "ext-{},ext-{}\n".format(_image, _label)
                        f.write(_str)
        return

    def do(self):
        if global_queue.remote_extend_queue.empty():
            return

        while not global_queue.remote_extend_queue.empty():
            _task = global_queue.remote_extend_queue.get()

            if not isinstance(_task, RemoteTask):
                break

            if _task.exit_flag:
                break

            src_path = _task.src_path
            dest_path = _task.dest_path

            _image = cv2.imread(src_path)
            if _image is None:
                print(src_path)
            else:
                width = _image.shape[1]
                height = _image.shape[0]

                # 只取图片的下半部分
                crop_img = _image[int(height // 2):height, 0:width]

                new_w = width + self.pixel * 2
                new_h = int(height // 2) + self.pixel * 2

                blank_image = np.zeros((new_h, new_w, 3), np.uint8)
                blank_image[0:new_h, 0:new_w] = (255, 255, 255)
                blank_image[self.pixel:new_h - self.pixel, self.pixel:new_w - self.pixel] = crop_img
                cv2.imwrite(dest_path, blank_image)

        return
