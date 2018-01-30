# -*-coding:utf-8-*-

import os
import time
import cv2
import json
import numpy as np

import multiprocessing

from mask_color import mask_labels


def do(dir_id):
    time_start = time.time()
    json_file = os.path.join(label_dir, str(dir_id), str(dir_id) + ".json")

    with open(json_file, "r") as f:
        json_data = json.load(f)

        if json_data is None:
            exit(0)

    images = json_data["imgs"]
    for img_id, img in images.items():
        # time1 = time.time()

        image_name = img["path"]
        label_name = "label-" + image_name.split(".")[0] + ".png"

        image_path = os.path.join(label_dir, str(dir_id), image_name)
        label_path = os.path.join(label_dir, str(dir_id), label_name)

        image = cv2.imread(image_path)
        width = image.shape[1]
        height = image.shape[0]

        blank_image = np.zeros((height, width, 3), np.uint8)
        blank_image[0:height, 0:width] = (other_clr[2], other_clr[1], other_clr[0])

        # [category, bbox]
        label_objects = img["objects"]

        ind = 0
        area_list = {}
        for rect_info in label_objects:
            xmin = int(rect_info["bbox"]["xmin"])
            ymin = int(rect_info["bbox"]["ymin"])
            xmax = int(rect_info["bbox"]["xmax"])
            ymax = int(rect_info["bbox"]["ymax"])
            area_list[ind] = (xmax - xmin) * (ymax - ymin)
            ind += 1

        dict = sorted(area_list.items(), key=lambda d: d[1], reverse=False)

        shape_list = []
        for ind, area in dict:
            shape_list.append(label_objects[ind])

        for rect_info in shape_list:
            object_id = rect_info["id"]
            _id = str(rect_info["category"])
            if _id.startswith("i"):
                _id = "0"
            elif _id.startswith("p"):
                _id = "1"
            elif _id.startswith("w"):
                _id = "2"

            xmin = int(rect_info["bbox"]["xmin"])
            ymin = int(rect_info["bbox"]["ymin"])
            xmax = int(rect_info["bbox"]["xmax"])
            ymax = int(rect_info["bbox"]["ymax"])

            _cls_id = int(_id)
            _cls_color = mask_colors[_cls_id]

            # cv2.rectangle(blank_image, (xmin, ymin), (xmax, ymax), (0, 0, 0), 3)

            has_parent = False
            object_count = len(shape_list)
            for i in range(object_count):
                _rect_info = shape_list[i]
                if _rect_info["id"] == object_id:
                    continue

                _xmin = int(_rect_info["bbox"]["xmin"])
                _ymin = int(_rect_info["bbox"]["ymin"])
                _xmax = int(_rect_info["bbox"]["xmax"])
                _ymax = int(_rect_info["bbox"]["ymax"])

                if xmin >= _xmin and xmax <= _xmax and ymin >= _ymin and ymax <= _ymax:
                    has_parent = True
                    break

            if has_parent:
                continue

            if "polygon" in rect_info:
                pts = rect_info["polygon"]

                polygon = []
                pt_count = int(len(pts) / 2)
                if pt_count > 0:
                    tmp = pts[0]
                    if isinstance(tmp, list):
                        pt_count = len(pts)
                        for i in range(pt_count):
                            x, y = pts[i]
                            polygon.append([int(x), int(y)])
                    else:
                        for i in range(pt_count):
                            x = pts[i * 2]
                            y = pts[i * 2 + 1]
                            polygon.append([int(x), int(y)])
                draw_pts = np.array(polygon)
                draw_clr = (_cls_color[2], _cls_color[1], _cls_color[0])
                cv2.fillPoly(img=blank_image, pts=[draw_pts], color=draw_clr)

        cv2.imwrite(label_path, blank_image)
        # time2 = time.time()
        # print("process [{}] in {} s".format(image_name, time2-time1))

    time_end = time.time()
    print("process [{}] in {} s".format(dir_id, time_end - time_start))

    print("start to make annotation....")
    time_start = time.time()
    json_dir = os.path.dirname(json_file)

    file_list = os.listdir(json_dir)
    for file_id in file_list:
        # time1 = time.time()
        if not file_id.startswith("label-"):
            continue

        annotation = file_id[6:]
        annotation_path = os.path.join(json_dir, annotation)

        image_path = os.path.join(json_dir, file_id)
        image = cv2.imread(image_path)
        width = image.shape[1]
        height = image.shape[0]

        label_data = np.zeros((height, width), np.uint8)
        label_data[0:height, 0:width] = other_id

        for label in mask_labels:
            color = (label.color[2], label.color[1], label.color[0])
            label_data[np.where((image == color).all(axis=2))] = label.categoryId

        cv2.imwrite(annotation_path, label_data)
        # time2 = time.time()
        # print("process [{}] in {} s".format(file_id, time2-time1))

    time_end = time.time()
    print("process [{}] in {} s".format(dir_id, time_end - time_start))

if __name__ == "__main__":

    label_dir = "/data/deeplearning/dataset/training/sign_mask"
    # label_dir = "/Users/zhangbenxing/check/sign_mask"
    max_packages = 1000000

    mask_colors = {label.categoryId: label.color for label in mask_labels}

    other_clr = (0, 0, 0)
    other_id = -1
    for label in mask_labels:
        if label.name == u"其它":
            other_clr = label.color
            other_id = label.categoryId
            break

    all_process = []
    for dir_id in range(max_packages):
        json_file = os.path.join(label_dir, str(dir_id), str(dir_id) + ".json")
        if not os.path.exists(json_file):
            continue

        process = multiprocessing.Process(target=do, args=(dir_id,))
        all_process.append(process)

    for process in all_process:
        process.start()

    for process in all_process:
        process.join()

