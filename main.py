# -*- coding:utf-8 -*-
import sys
sys.path.insert(0, "/opt/auto_training")
import errno
import json
import logging
import os
import shutil
import signal

import tornado.ioloop
import tornado.web

from road_marking.model_release import ModelReleaseHandler
from road_marking.process_label import ProcessLabelHandler
from road_marking.task_divide import TaskDivideHandler
from road_marking.training import StartTrainingHandler
from road_marking.check_label import CheckLabelHandler

from utils import host_ip, port


class MainHandler(tornado.web.RequestHandler):
    def get(self, *args, **kwargs):
        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Content-type', 'application/json')

        try:
            src_dir = "/data/deeplearning/dataset/training/models/released"
            dest_dir = "/data/deeplearning/dataset/training/models/prod"

            err_code = "0"
            err_info = "success"

            if os.path.exists(src_dir):
                model_file = os.listdir(src_dir)
                model_exist = False
                for model in model_file:
                    name_list = str(model).split(".")
                    if len(name_list) == 2:
                        name_ext = name_list[1]
                        if name_ext == "params":
                            model_exist = True
                            src_path = os.path.join(src_dir, model)
                            dest_path = os.path.join(dest_dir, "camvid_decoder2-kd_cls6_s2_ep-0200.params")
                            shutil.copyfile(src_path, dest_path)
                        elif name_ext == "json":
                            src_path = os.path.join(src_dir, model)
                            dest_path = os.path.join(dest_dir, "camvid_decoder2-kd_cls9_s2_ep-symbol.json")
                            shutil.copyfile(src_path, dest_path)
                if not model_exist:
                    err_code = "2"
                    err_info = "none model file exist"
            else:
                err_code = "1"
                err_info = "/data/deeplearning/dataset/training/models/released is not exist"

            json_res = {"code": err_code, "msg": err_info}
            res_str = json.dumps(json_res)
            self.write(res_str)
        except Exception as err:
            err_info = err.args[0]
            json_res = {"code": "99", "msg": str(err_info)}
            self.write(json.dumps(json_res))
        except:
            self.write('{"code": "99", "msg": "unknown exception"}')

        self.finish()


def make_app():
    # generate_task = r"/{}/task".format(context)
    # process_label = r"/{}/label".format(context)
    # start_training = r"/{}/training".format(context)
    # release = r"/{}/release".format(context)
    return tornado.web.Application([
        (r"/update", MainHandler),
        (r"/task", TaskDivideHandler),
        (r"/label", ProcessLabelHandler),
        (r"/check", CheckLabelHandler),
        (r"/training", StartTrainingHandler),
        (r"/release", ModelReleaseHandler),
    ])


def wait_child(signum, frame):
    print('receive SIGCHLD')
    try:
        while True:
            cpid, status = os.waitpid(-1, os.WNOHANG)
            if cpid == 0:
                print('no child process was immediately available')
                break
            exitcode = status >> 8
            print('child process %s exit with exitcode %s', cpid, exitcode)
    except OSError as e:
        if e.errno == errno.ECHILD:
            print('current process has no existing unwaited-for child processes.')
        else:
            raise
    print('handle SIGCHLD end')

if __name__ == "__main__":
    signal.signal(signal.SIGCHLD, wait_child)

    app = make_app()
    try:
        app.listen(port)
        print("server start with[{}:{}]".format(host_ip, port))
    except Exception as e:
        print("start error:{}".format(repr(e)))
        exit(0)

    logger = logging.getLogger("auto-training")
    cur_path = os.path.realpath(__file__)
    cur_dir = os.path.dirname(cur_path)
    log_file = os.path.join(cur_dir, "logs", "auto-training.log")

    formatter = logging.Formatter('%(asctime)s -%(name)s-%(levelname)s-%(module)s:%(message)s')
    # 文件日志
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)  # 可以通过setFormatter指定输出格式
    logger.addHandler(file_handler)

    # 指定日志的最低输出级别，默认为WARN级别
    logger.setLevel(logging.WARNING)

    tornado.ioloop.IOLoop.current().start()