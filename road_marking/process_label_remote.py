
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
from stat import S_ISDIR

import numpy as np

from utils import ServerResponse
from utils import create_ssh_client
from utils import host_ip, max_packages
from utils import Task
from label import self_road_chn_labels

import global_queue


class ProcessLabelRemoteHandler(tornado.web.RequestHandler):
    def initialize(self):
        self.file_list = list()
        self.pixel = 50

        self.src_dir = "/data/deeplearning/dataset/training/data/released"
        self.temp_dir = "/data/deeplearning/dataset/training/data/released_temp"
        self.dest_dir = "/data/deeplearning/dataset/kd/lane"

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

        try:
            _ver = self.get_argument("version", "all")
            self.src_dir = os.path.join(self.src_dir, _ver)
            self.temp_dir = os.path.join(self.temp_dir, _ver)

            if not os.path.exists(self.src_dir):
                task_count = 0
                err_code = 1
            else:
                if not os.path.exists(self.temp_dir):
                    os.makedirs(self.temp_dir)
                # else:
                #     # clean this directory
                #     tmp_dirs = os.listdir(self.temp_dir)
                #     for tmp_dir in tmp_dirs:
                #         if not tmp_dir.isdigit():
                #             continue
                #         tmp_path = os.path.join(self.temp_dir, tmp_dir)
                #         if os.path.isfile(tmp_path):
                #             os.remove(tmp_path)
                #         else:
                #             shutil.rmtree(tmp_path)

                err_code = 0
                for dir in range(max_packages):
                    csv_name = "ext-" + str(dir) + ".csv"
                    csv_file = os.path.join(self.src_dir, str(dir), csv_name)
                    if not os.path.exists(csv_file):
                        continue

                    self.clean_dir(_csv_file=csv_file)
                    self.prepare_crop(_csv_file=csv_file)

                task_count = global_queue.remote_cut_queue.qsize()
                task_count = task_count // 2
                # start to process
                task = Task(None, None, None, None, True)
                global_queue.remote_cut_queue.put(task)

                self.do_work()

                for i in range(max_packages):
                    _sub_dir = os.path.join(self.src_dir, str(i))
                    if not os.path.exists(_sub_dir):
                        continue

                    dest_dir = os.path.join(self.temp_dir, str(i))
                    if not os.path.exists(dest_dir):
                        os.makedirs(dest_dir)

                    origin_list = os.listdir(_sub_dir)

                    for _image in origin_list:
                        _image = _image.strip()
                        if not _image.startswith("label-"):
                            continue

                        if not _image.endswith("png"):
                            continue

                        # start with label-
                        anna_file = _image[6:]
                        origin_name = anna_file[:-3] + "jpg"
                        image_path = os.path.join(_sub_dir, _image)
                        origin_image = os.path.join(_sub_dir, origin_name)
                        if not os.path.exists(origin_image):
                            os.remove(image_path)
                            self.logger.error("package:{}, file missing:[{}]=>[{}]".format(str(i), origin_name, _image))
                            continue

                        result_path = os.path.join(self.temp_dir, str(i), anna_file)

                        task = Task(str(i), image_path, result_path, origin_image, False)
                        global_queue.remote_process_queue.put(task)
                for i in range(20):
                    task = Task(None, None, None, None, True)
                    global_queue.remote_process_queue.put(task)

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
                if _ver != "all":
                    cur_day = time.strftime("lane-aug-%Y%m%d", time.localtime())
                else:
                    cur_day = time.strftime("lane-all-%Y%m%d", time.localtime())
                copy_dir = self.temp_dir
                dir_list = os.listdir(copy_dir)

                if self.dest_scp_ip != host_ip:
                    files = self.dest_sftp.listdir(path=self.dest_dir)
                    self.dest_dir = os.path.join(self.dest_dir, cur_day)
                    # if cur_day in files:
                    #     self.rm(self.dest_dir)
                    if cur_day not in files:
                        self.dest_sftp.mkdir(self.dest_dir)

                for _dir in dir_list:
                    old_src = os.path.join(copy_dir, _dir)
                    files = self.dest_sftp.listdir(path=self.dest_dir)
                    if _dir in files:
                        self.rm(os.path.join(self.dest_dir, _dir))
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

    def prepare_crop(self, _csv_file):
        root_dir = os.path.dirname(_csv_file)
        csv_name = os.path.basename(_csv_file)

        name_list = csv_name.split(".")
        csv_name = name_list[0]
        if csv_name.startswith("ext-"):
            csv_name = csv_name[4:]
        package_index = csv_name

        with open(_csv_file, "r") as f:
            line_str = f.readline()

            while line_str:
                image, label = line_str.split(",")
                image = image.strip()
                label = label.strip()

                if not image.startswith("ext-") or not label.startswith("ext-"):
                    line_str = f.readline()
                    continue

                if not os.path.exists(os.path.join(root_dir, label)):
                    line_str = f.readline()
                    self.logger.error("package:{}, label missing:[{}]=>[{}]".format(package_index, image, label))
                    continue

                if not os.path.exists(os.path.join(root_dir, image)):
                    line_str = f.readline()
                    self.logger.error("package:{}, image missing:[{}]=>[{}]".format(package_index, image, label))
                    continue

                origin_image = os.path.join(root_dir, image[4:])
                _src = os.path.join(root_dir, image)
                _dest_image = os.path.join(root_dir, origin_image)
                task = Task(package_index, _src, _dest_image, None)
                global_queue.remote_cut_queue.put(task)

                _src = os.path.join(root_dir, label)
                _dest_id2 = label[4:]
                _dest_label = os.path.join(root_dir, _dest_id2)
                _dest_label = _dest_label.strip()

                task = Task(package_index, _src, _dest_label, None)
                global_queue.remote_cut_queue.put(task)

                line_str = f.readline()

    def do_work(self):
        if global_queue.remote_cut_queue.empty():
            return

        while not global_queue.remote_cut_queue.empty():
            task = global_queue.remote_cut_queue.get()

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

            if width == 2448 and height == 1024:
                continue

            crop_img = img[self.pixel:height - self.pixel, self.pixel:width - self.pixel]
            cv2.imwrite(_dest, crop_img)

            time2 = time.time()
            self.logger.info("process[{}/{}] in {} s".format(task.package_index, _src, time2 - time1))

    def transform(self):
        while not global_queue.remote_process_queue.empty():
            task = global_queue.remote_process_queue.get()

            if not isinstance(task, Task):
                break

            if task.exit_flag:
                break

            image_path = task.src_path
            result_path = task.dest_path
            origin_image_path = task.dest_label

            time1 = time.time()

            img = cv2.imread(image_path)
            if img is None:
                self.logger.error("image is none[{}/{}]".format(task.package_index, image_path))
                continue

            width = img.shape[1]
            height = img.shape[0]

            other_category = 255
            label_data = np.zeros((height, width), np.uint8)
            for label in self_road_chn_labels:
                if label.name == u"Ignore":
                    other_category = label.categoryId
                    break
            label_data[0:height, 0:width] = other_category

            for label in self_road_chn_labels:
                color = (label.color[2], label.color[1], label.color[0])
                label_data[np.where((img == color).all(axis=2))] = label.categoryId

            # 校验"Ignore"类别的占比
            other_count = np.sum(label_data == other_category)
            valid_count = width*height*0.01
            if other_count > valid_count:
                label_name = os.path.basename(image_path)
                file_name = label_name.split(".")[0]
                if file_name.startswith("ext-"):
                    file_name = file_name[4:]
                if file_name.startswith("label-"):
                    file_name = file_name[6:]
                origin_image_path = os.path.join(os.path.dirname(image_path), file_name+".jpg")
                self.logger.error("label[{}/{}] not qualified".format(task.package_index, image_path))

                if os.path.exists(origin_image_path):
                    os.remove(origin_image_path)
                if os.path.exists(image_path):
                    os.remove(image_path)
            else:
                cv2.imwrite(result_path, label_data)
                dest_image_path = result_path[:-3] + "jpg"
                shutil.copy(origin_image_path, dest_image_path)
                dest_label_path = os.path.join(os.path.dirname(result_path), os.path.basename(image_path))
                shutil.copy(image_path, dest_label_path)

            time2 = time.time()
            self.logger.info("process[{}/{}] in {} s".format(task.package_index, image_path, time2 - time1))
        exit(0)

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

                        annotation_file = origin_image[:-3] + "png"
                        image_list.append(str(annotation_file).strip())

                    if label.startswith("ext-"):
                        origin_label = label[4:]
                        image_list.append(str(origin_label).strip())

                if not os.path.exists(image_path):
                    if os.path.exists(label_path):
                        self.logger.error("package:{}, not match:[{}] not exist=>[{}] exist".format(package_index, image, label))
                        if os.path.exists(label_path):
                            os.remove(label_path)
                    else:
                        self.logger.error("package:{}, file missing:[{}]=>[{}]".format(package_index, image, label))
                else:
                    if not os.path.exists(label_path):
                        self.logger.error("package:{}, not match:[{}] exist=>[{}] not exist".format(package_index, image, label))
                        if os.path.exists(image_path):
                            os.remove(image_path)

                line_str = f.readline()

        file_list = os.listdir(root_dir)
        for file_id in file_list:
            file_id = str(file_id).strip()

            if file_id.endswith("txt"):
                continue
            if file_id not in image_list:
                file_delete = os.path.join(root_dir, file_id)
                self.logger.error("package:{}, file wrong:[{}]".format(package_index, file_id))
                if os.path.exists(file_delete):
                    if os.path.isfile(file_delete):
                        os.remove(file_delete)
