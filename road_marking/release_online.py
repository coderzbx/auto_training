
# -*- coding:utf-8 -*-

import tornado.web
import os
import json
import time
import multiprocessing
import logging
import shutil

import cv2
import scp
import requests
from stat import S_ISDIR

import numpy as np

from utils import ServerResponse
from utils import create_ssh_client
from utils import host_ip
from utils import OnlineTask
from utils import self_road_chn_labels

import global_queue


class ReleaseOnlineHandler(tornado.web.RequestHandler):
    def initialize(self):
        self.file_list = list()
        self.pixel = 50

        self.src_dir = "/data/deeplearning/dataset/training/data/released"
        self.temp_dir = "/data/deeplearning/dataset/training/data/released_temp"
        self.dest_dir = "/data/deeplearning/dataset/kd/lane"

        # self.released_url = "http://192.168.5.31:23300/kts/runtime/tasks?"
        self.krs_url = "http://192.168.5.34:33100/krs/image/get?"
        self.released_url = "http://192.168.5.34:33300/kts"

        self.dest_scp_ip = "192.168.5.36"
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

        self.logger = logging.getLogger("auto-training")

    def __delete__(self, instance):
        if self.dest_scp:
            self.dest_scp.close()
        if self.dest_sftp:
            self.dest_sftp.close()
        if self.dest_ssh:
            self.dest_ssh.close()

    def isdir(self, path):
        try:
            return S_ISDIR(self.dest_sftp.stat(path).st_mode)
        except IOError:
            return False

    def rm(self, path):
        files = self.dest_sftp.listdir(path=path)

        for f in files:
            filepath = os.path.join(path, f)
            if self.isdir(filepath):
                self.rm(filepath)
            else:
                self.dest_sftp.remove(filepath)
        self.dest_sftp.rmdir(path)

    def get(self, *args, **kwargs):
        time1 = time.time()

        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Content-type', 'application/json')

        # {
        #     “images”:”原始图片目录，缺省默认为当前目录的images目录”,
        #     “pixel”:外边框像素大小，缺省默认为50,
        # }
        err_code = 0
        try:
            cur_day = time.strftime("%Y%m%d", time.localtime())
            self.src_dir = os.path.join(self.src_dir, cur_day, "1")
            self.temp_dir = os.path.join(self.temp_dir, cur_day, "1")

            if not os.path.exists(self.src_dir):
                os.makedirs(self.src_dir)
            else:
                if not os.path.exists(self.temp_dir):
                    os.makedirs(self.temp_dir)
                else:
                    # clean this directory
                    tmp_dirs = os.listdir(self.temp_dir)
                    for tmp_dir in tmp_dirs:
                        if not tmp_dir.isdigit():
                            continue
                        tmp_path = os.path.join(self.temp_dir, tmp_dir)
                        if os.path.isfile(tmp_path):
                            os.remove(tmp_path)
                        else:
                            shutil.rmtree(tmp_path)

            # download released work
            release_url = "{}/mark/preSubmit".format(self.released_url)
            resp = requests.get(url=release_url)
            if resp.status_code != 200:
                raise Exception("get released task receive status[{}]".format(resp.status_code))

            json_data = json.loads(resp.text)
            released_tasks = None
            if "code" in json_data:
                code = json_data["code"]
                if code != "0":
                    if "message" in json_data:
                        msg = json_data["message"]
                    else:
                        msg = "code={}".format(code)
                    raise Exception(msg)
                else:
                    if "result" in json_data:
                        released_tasks = json_data["result"]["data"]
                        if len(released_tasks) == 0:
                            raise Exception("released task count is 0")
            elif "data" in json_data:
                released_tasks = json_data["data"]
            if released_tasks is None:
                raise Exception("can't find data")

            task_count = len(released_tasks)

            track_point_id = None
            image_type = "70"
            image_seq = "000"
            image_suffix = "jpg"
            for task in released_tasks:
                task_id = task["id"]
                variables = task["variables"]
                for variable in variables:
                    if variable["name"] == "TRACKPOINTID":
                        track_point_id = variable["value"]
                        break
                down_image_url = "{}trackPointId={}&type={}&seq={}&imageType={}".format(
                    self.krs_url, track_point_id, image_type, image_seq, image_suffix
                )
                resp = requests.get(url=down_image_url)
                if resp.status_code != 200:
                    raise Exception("download image receive status[{}]".format(resp.status_code))
                content_type = resp.headers['Content-Type']
                if not str(content_type).startswith("image"):
                    raise Exception(resp.text)

                image_data = resp.content

                image_seq = "001"
                image_suffix = "png"
                down_label_url = "{}trackPointId={}&type={}&seq={}&imageType={}".format(
                    self.krs_url, track_point_id, image_type, image_seq, image_suffix
                )
                resp = requests.get(url=down_label_url)
                if resp.status_code != 200:
                    raise Exception("download label receive status[{}]".format(resp.status_code))
                content_type = resp.headers['Content-Type']
                if not str(content_type).startswith("image"):
                    raise Exception(resp.text)
                label_data = resp.content

                _task = OnlineTask(
                    track_point_id=track_point_id, task_id=task_id, image_data=image_data, label_data=label_data
                )
                global_queue.online_queue.put(_task)

            for i in range(20):
                task = OnlineTask(None, None, None, None, exit_flag=True)
                global_queue.online_queue.put(task)

            all_processes = []
            for i in range(20):
                process = multiprocessing.Process(target=self.transform)
                process.daemon = True
                all_processes.append(process)

            for process in all_processes:
                process.start()
                self.logger.info(str(process.pid) + ", start")

            for process in all_processes:
                process.join()
                self.logger.info(str(process.pid) + ", join")

            # 拷贝
            copy_dir = os.path.dirname(self.temp_dir)
            dir_list = os.listdir(copy_dir)
            remote_day = time.strftime("lane-aug-%Y%m%d", time.localtime())

            if self.dest_scp_ip != host_ip:
                files = self.dest_sftp.listdir(path=self.dest_dir)
                self.dest_dir = os.path.join(self.dest_dir, remote_day)
                if remote_day in files:
                    self.rm(self.dest_dir)
                self.dest_sftp.mkdir(self.dest_dir)

            for _dir in dir_list:
                old_src = os.path.join(copy_dir, _dir)
                self.dest_scp.put(old_src, self.dest_dir, recursive=True)

            time2 = time.time()
            result_obj = {
                "count": str(task_count),
                "time": str(time2-time1)+" s"
            }
            resp = ServerResponse(err_code=err_code, err_info=None, result=result_obj)
            resp_str = resp.generate_response()
            self.logger.info(resp_str)

            self.write(resp_str)
        except Exception as err:
            err_info = repr(err)
            json_res = {"code": "99", "msg": str(err_info)}
            self.logger.error(json.dumps(json_res))
            self.write(json.dumps(json_res))

        self.finish()

    def transform(self):
        while not global_queue.online_queue.empty():
            task = global_queue.online_queue.get()

            if not isinstance(task, OnlineTask):
                break

            if task.exit_flag:
                break

            time1 = time.time()

            image_data = task.image_data
            label_data = task.label_data

            # status： 0 成功 1 图片不通过 2 缺少两张图片 3 缺少原始图片 4缺少标注图片
            err_code = 0

            try:
                _image0 = np.asarray(bytearray(image_data), dtype="uint8")
                _image = cv2.imdecode(_image0, cv2.IMREAD_COLOR)

                _label0 = np.asarray(bytearray(label_data), dtype="uint8")
                _label = cv2.imdecode(_label0, cv2.IMREAD_COLOR)

                if _image is None:
                    self.logger.error("image is none[{}]".format(task.task_id))
                    err_code = 3
                    raise Exception("image is none[{}]".format(task.task_id))
                if _label is None:
                    self.logger.error("label is none[{}]".format(task.task_id))
                    err_code = 4
                    raise Exception("label is none[{}]".format(task.task_id))

                width = _image.shape[1]
                height = _image.shape[0]

                width_ = _label.shape[1]
                height_ = _label.shape[0]
                if width != width_ or height != height_:
                    self.logger.error("{} size is not equal".format(task.task_id))
                    err_code = 1
                    raise Exception("{} size is not equal".format(task.task_id))

                # save file
                image_path = os.path.join(self.src_dir, "{}.jpg".format(task.track_point_id))
                label_path = os.path.join(self.src_dir, "label-{}.png".format(task.track_point_id))

                cv2.imwrite(image_path, _image)
                cv2.imwrite(label_path, _label)

                # other_category = -1
                instance_data = np.zeros((height, width), np.uint8)
                instance_data[0:height, 0:width] = 255
                # for label in self_road_chn_labels:
                #     if label.name == u"其他":
                #         other_category = label.categoryId
                #         break

                for label in self_road_chn_labels:
                    color = (label.color[2], label.color[1], label.color[0])
                    instance_data[np.where((_label == color).all(axis=2))] = label.categoryId

                # 校验"其他"类别的占比
                other_count = np.sum(instance_data == 255)
                valid_count = width * height * 0.01
                # if other_count > valid_count:
                #     self.logger.error("label[{}/{}] not qualified".format(task.task_id, task.track_point_id))
                #     err_code = 1
                #     raise Exception("label[{}/{}] not qualified".format(task.task_id, task.track_point_id))

                image_path1 = os.path.join(self.temp_dir, "{}.jpg".format(task.track_point_id))
                label_path1 = os.path.join(self.temp_dir, "label-{}.png".format(task.track_point_id))
                shutil.copy(image_path, image_path1)
                shutil.copy(label_path, label_path1)

                instance_path = os.path.join(self.temp_dir, "{}.png".format(task.track_point_id))
                cv2.imwrite(instance_path, instance_data)

                # callback
                callback_url = "{}/mark/commit".format(self.released_url)
                post_data = {
                    "claimId": task.task_id,
                    "status": str(err_code)
                }
                resp = requests.post(url=callback_url, data=json.dumps(post_data))
                if resp.status_code != 200:
                    self.logger.error("{} callback error:{}".format(task.task_id, resp.text))

                time2 = time.time()
                self.logger.info("process[{}/{}] in {} s".format(task.task_id, task.track_point_id, time2 - time1))
            except Exception as e:
                callback_url = "{}/mark/commit".format(self.released_url)
                post_data = {
                    "claimId": task.task_id,
                    "status": str(err_code)
                }
                resp = requests.post(url=callback_url, data=json.dumps(post_data))
                if resp.status_code != 200:
                    self.logger.error("{} callback error:{}".format(task.task_id, resp.text))

                self.logger.error(repr(e))
        exit(0)
