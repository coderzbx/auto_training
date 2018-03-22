import os
import time
import argparse
import json
import multiprocessing

import requests
import cv2

from model import ModelMaskrcnn
from labels import kd_traffic_sign_labels


class Task:
    def __init__(self, track_id, track_point_id, dest_dir):
        self.track_id = track_id
        self.track_point_id = track_point_id
        self.dest_dir = dest_dir


class FetchImage:
    def __init__(self, krs, track_ids, dest_dir, manager):
        self.krs_url = krs
        self.track_ids = track_ids
        self.save_dir = dest_dir

        self.id_name = {label.trainId: label.alias for label in kd_traffic_sign_labels}

        self.manager = manager
        self.task_queue = manager.Queue()

    def prepare(self):
        url = krs_url + "/track/get"
        for track_id in self.track_ids:
            res = requests.post(url=url, data={'trackId': track_id})
            track_info = res.text
            track_data = json.loads(track_info)
            code = track_data["code"]

            if code != "0":
                print(track_info.encode("utf8"))
                continue

            points = track_data["result"]["pointList"]
            track_list = []
            for track_point_id in points:
                track_list.append(track_point_id["trackPointId"])
            track_list.sort()
            for track_point_id in track_list:
                task = Task(track_id=track_id, track_point_id=track_point_id, dest_dir=self.save_dir)
                self.task_queue.put(task)

    def do(self, gpu_id):
        model = ModelMaskrcnn(gpu_id=gpu_id)

        while not self.task_queue.empty():
            task = self.task_queue.get()
            if not isinstance(task, Task):
                continue

            # download image
            url = self.krs_url + "/image/get"
            data = {
                "trackPointId": task.track_point_id,
                "type": "00",
                "seq": "004",
                "imageType": "jpg"
            }
            try:
                _start = time.time()
                res_data = requests.post(url=url, data=data)
                image_data = res_data.content
                content_type = res_data.headers['Content-Type']
                if not str(content_type).startswith("image"):
                    print(image_data)
                    continue

                instance_bbox = model.do(image_data=image_data)
                _end = time.time()
                print("processed {} in {} s".format(task.track_point_id, _end - _start))
                if len(instance_bbox) > 0:
                    find_sign = False
                    for instance_id, (bbox, score) in instance_bbox.items():
                        class_id = instance_id // 1000
                        class_name = self.id_name[int(class_id)]
                        if class_name in ["warning", "prohibition", "instructive"]:
                            find_sign = True
                            break
                    if not find_sign:
                        continue

                    # save image
                    track_dir = os.path.join(task.dest_dir, task.track_id, "images")
                    if not os.path.exists(track_dir):
                        os.makedirs(track_dir)
                    image_name = "{}_{}_{}.{}".format(task.track_point_id, "00", "004", "jpg")
                    dest_path = os.path.join(track_dir, image_name)
                    with open(dest_path, "wb") as _f:
                        _f.write(image_data)

                    # get bbox
                    image = cv2.imread(dest_path)
                    for instance_id, (bbox, score) in instance_bbox.items():
                        class_id = instance_id // 1000
                        class_name = self.id_name[int(class_id)]
                        if class_name not in ["warning", "prohibition", "instructive"]:
                            continue

                        index_id = instance_id % 1000
                        print("{}:class:{},index:{},score:{},bbox:{}".format(
                            task.track_point_id, class_name, index_id, score, bbox
                        ))

                        new_image = image[bbox[1]: bbox[3], bbox[0]: bbox[2]]
                        new_dir = os.path.join(task.dest_dir, task.track_id, "labels", class_name)
                        if not os.path.exists(new_dir):
                            os.makedirs(new_dir)
                        new_path = "{}_{}_{}.jpg".format(task.track_point_id, class_id, index_id)
                        new_path = os.path.join(new_dir, new_path)
                        cv2.imwrite(new_path, new_image)

                    for instance_id, (bbox, score) in instance_bbox.items():
                        class_id = instance_id // 1000
                        class_name = self.id_name[int(class_id)]
                        if class_name not in ["warning", "prohibition", "instructive"]:
                            continue
                        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), thickness=3)
                        cv2.putText(image, class_name, (bbox[2], bbox[3]),
                                    fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=2, color=(0, 255, 0), thickness=1)

                    sign_path = os.path.join(task.dest_dir, task.track_id, "labels", os.path.basename(dest_path))
                    cv2.imwrite(sign_path, image)

            except Exception as e:
                print(e.args[0])


if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--krs', type=str, default="31", required=False)
    parser.add_argument('--file', type=str, required=False)
    parser.add_argument('--trackId', type=str, required=False)
    parser.add_argument('--gpus', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)
    _params = parser.parse_args()

    save_dir = _params.dest_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    krs_url = "http://192.168.5.31:23100/krs"
    if _params.krs:
        if _params.krs == "30":
            krs_url = "http://192.168.5.30:23100/krs"
        elif _params.krs == "34":
            krs_url = "http://192.168.5.34:33100/krs"

    trackIds = []
    if _params.file and os.path.exists(_params.file):
        with open(_params.file, "r") as f:
            line_str = f.readline()
            while line_str:
                line_str = line_str.strip()
                line_str = line_str.strip("\"")
                trackIds.append(line_str)
                line_str = f.readline()

    if _params.trackId:
        trackIds = str(_params.trackId).split(",")

    fetch_handler = FetchImage(krs=krs_url, track_ids=trackIds, dest_dir=save_dir, manager=multiprocessing.Manager())
    fetch_handler.prepare()

    gpus = _params.gpus
    gpus = str(gpus).split(",")

    all_process = []
    for gpu_id in gpus:
        processor = multiprocessing.Process(target=fetch_handler.do, args=(int(gpu_id),))
        all_process.append(processor)

    for proc_ in all_process:
        proc_.start()

    for proc_ in all_process:
        proc_.join()

    end = time.time()

    print("finish in {} s".format(end - start))
