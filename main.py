# -*- coding:utf-8 -*-
import sys
sys.path.insert(0, "/opt/auto_training")
import errno
import json
import logging
import os
import signal

import tornado.ioloop
import tornado.web

from road_marking.release_online import ReleaseOnlineHandler
from road_marking.process_label import ProcessLabelHandler
from road_marking.process_label_remote import ProcessLabelRemoteHandler
from road_marking.process_label_local import ProcessLabelLocalHandler
from road_marking.task_divide import TaskDivideHandler
from road_marking.task_divide_remote import TaskDivideRemoteHandler
from road_marking.check_label import CheckLabelHandler

import global_variables


class MainHandler(tornado.web.RequestHandler):
    def get(self, *args, **kwargs):
        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Content-type', 'application/json')

        try:
            err_code = "0"
            err_info = "success"

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


class IconHandler(tornado.web.RequestHandler):
    def get(self, *args, **kwargs):
        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Content-type', 'application/json')

        self.write('auto-training')
        self.finish()


def make_app():
    # generate_task = r"/{}/task".format(context)
    # process_label = r"/{}/label".format(context)
    # start_training = r"/{}/training".format(context)
    # release = r"/{}/release".format(context)
    return tornado.web.Application([
        (r"/favicon.ico", IconHandler),
        (r"/update", MainHandler),
        (r"/task", TaskDivideHandler),
        (r"/task/remote", TaskDivideRemoteHandler),
        (r"/label", ProcessLabelHandler),
        (r"/label/remote", ProcessLabelRemoteHandler),
        (r"/label/local", ProcessLabelLocalHandler),
        (r"/check", CheckLabelHandler),
        (r"/release/online", ReleaseOnlineHandler),
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
        app.listen(global_variables.port.value)
        print("server start with[{}:{}]".format(global_variables.host_ip.value,
                                                global_variables.port.value))
    except Exception as e:
        print("start error:{}".format(repr(e)))
        exit(0)

    logger = logging.getLogger("auto-training")
    cur_path = os.path.realpath(__file__)
    cur_dir = os.path.dirname(cur_path)
    log_file = os.path.join(cur_dir, "logs", "auto-training.log")

    cur_path = os.path.realpath(__file__)
    cur_dir = os.path.dirname(cur_path)
    config_path = os.path.join(cur_dir, "recognition.conf")
    if not os.path.exists(config_path):
        config_path = None

    if config_path is None or not os.path.exists(config_path):
        print ("configure file server failed")
        exit(0)

    if not global_variables.init_params(config_path=config_path):
        print ("server initial failed")
        exit(0)

    formatter = logging.Formatter('%(asctime)s -%(name)s-%(levelname)s-%(module)s:%(message)s')
    # 文件日志
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)  # 可以通过setFormatter指定输出格式
    logger.addHandler(file_handler)

    # 指定日志的最低输出级别，默认为DEBUG级别
    logger.setLevel(logging.DEBUG)

    tornado.ioloop.IOLoop.current().start()