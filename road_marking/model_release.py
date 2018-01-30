# -*- coding:utf-8 -*-

import tornado.web
import os
import json
import shutil
import multiprocessing

import scp

from utils import ServerResponse
from utils import create_ssh_client
from utils import host_ip


class ModelReleaseHandler(tornado.web.RequestHandler):
    def initialize(self):
        self.task_queue = multiprocessing.Manager().Queue()
        self.queue = multiprocessing.Manager().Queue()
        self.file_list = list()
        self.src_dir = "/data/deeplearning/dataset/training/models/released"
        self.dest_dir = "/data/deeplearning/dataset/training/models/prod"

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
        if self.dest_scp:
            self.dest_scp.close()
        if self.dest_sftp:
            self.dest_sftp.close()
        if self.dest_ssh:
            self.dest_ssh.close()

    def get(self, *args, **kwargs):
        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Content-type', 'application/json')

        try:
            if self.src_scp_ip != host_ip:
                # 拷贝文件
                if not os.path.exists(self.src_dir):
                    os.makedirs(self.src_dir)
                src_list = self.src_sftp.listdir(self.src_dir)
                for src_file in src_list:
                    src_path = os.path.join(self.src_dir, src_file)
                    self.src_scp.get(src_path, src_path)

            if not os.path.exists(self.src_dir):
                err_code = 1
                resp = ServerResponse(err_code=err_code, err_info=None, result=None)
                resp_str = resp.generate_response()
            else:
                # start to copy
                file_list = os.listdir(self.src_dir)
                result_obj = {
                    "files": file_list
                }
                resp = ServerResponse(err_code=0, err_info=None, result=result_obj)
                resp_str = resp.generate_response()

                for file_id in file_list:
                    src_path = os.path.join(self.src_dir, file_id)
                    if os.path.isfile(src_path):
                        dest_path = os.path.join(self.dest_dir, file_id)

                        if self.dest_scp_ip != host_ip:
                            self.dest_scp.put(src_path, dest_path)
                        else:
                            shutil.copy(src_path, dest_path)

            self.write(resp_str)
        except Exception as err:
            err_info = err.args[0]
            json_res = {"code": "99", "msg": str(err_info)}
            self.write(json.dumps(json_res))
        except:
            self.write('{"code": "99", "msg": "unknown exception"}')

        self.finish()
