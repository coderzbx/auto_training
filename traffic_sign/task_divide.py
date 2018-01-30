# -*- coding:utf-8 -*-

import tornado.web
import os
import json
import time
import multiprocessing
import logging
import shutil
import scp

from utils import ServerResponse
from utils import create_ssh_client
from utils import host_ip


class SignTaskDivideHandler(tornado.web.RequestHandler):
    def initialize(self):
        self.task_queue = multiprocessing.Manager().Queue()
        self.queue = multiprocessing.Manager().Queue()
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

                dir_content = os.listdir(self.src_dir)
                task_count = len(dir_content)

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
        img_list = []
        dir_content = os.listdir(src_dir)
        for id_ in dir_content:
            name_list = id_.split(".")
            if len(name_list) != 2:
                continue

            ext_name = name_list[1]
            if ext_name != 'png' and ext_name != 'jpg':
                continue
            img_list.append(id_)

        img_count = len(img_list)
        img_list.sort()

        result_objects = {}
        for id_ in img_list:
            file_image = id_.split("/")
            file_image = file_image[len(file_image) - 1]
            file_id = (file_image.split("."))[0]
            result_objects[file_id] = {"objects": [], "path": id_}

        annations = {
            "imgs": result_objects,
            "types": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
        }
        dest_file = os.path.join(dest_dir, "total_annotations.json")
        with open(dest_file, "w") as f:
            json.dump(annations, f)

        step = cnt_per_package
        imgs = annations["imgs"]

        # 数据包从1开始
        total_index = 0
        package_index = start_package
        package_annotations = {"types": annations["types"]}
        package_imgs = {}

        # step个数图片一个包
        image_index = 0

        package_dir = os.path.join(dest_dir, str(package_index))
        if not os.path.exists(package_dir):
            os.makedirs(package_dir)
        for img_id, img_info in imgs.items():
            image_index += 1

            path = img_info["path"]
            src_image = os.path.join(src_dir, path)

            path_list = path.split("/")
            image_name = path_list[len(path_list) - 1]

            dest_image = os.path.join(package_dir, image_name)
            shutil.copy(src_image, dest_image)

            package_imgs[img_id] = {"path": image_name, "objects": img_info["objects"]}

            if image_index == step:
                package_annotations["imgs"] = package_imgs
                annot_name = str(package_index) + ".json"
                annot_file = os.path.join(package_dir, annot_name)

                with open(annot_file, "w") as f:
                    json.dump(package_annotations, f)

                package_index += 1
                package_imgs = {}
                image_index = 0
                package_dir = os.path.join(dest_dir, str(package_index))
                if not os.path.exists(package_dir):
                    os.makedirs(package_dir)

            total_index += 1
            if total_index == img_count:
                package_annotations["imgs"] = package_imgs
                annot_name = str(package_index) + ".json"
                annot_file = os.path.join(package_dir, annot_name)

                with open(annot_file, "w") as f:
                    json.dump(package_annotations, f)
