import os
import time
import argparse
import multiprocessing

import cv2
import numpy as np
import random


if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)
    _params = parser.parse_args()

    save_dir = _params.dest_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not os.path.exists(_params.dir):
        exit(0)

    start_package = 1
    package_size = 20
    pixel = 50

    task_list = []
    file_list = os.listdir(_params.dir)
    for file_id in file_list:
        if file_id.endswith("jpg"):
            task_list.append(file_id)

    random.seed(10)
    random.shuffle(file_list)

    task_count = len(file_list)
    package_count = task_count // 20
    if task_count % 20 != 0:
        package_count += 1

    for package_id in range(package_count):
        if package_id == package_count - 1:
            package_list = file_list[package_id * package_size:]
        else:
            package_list = file_list[package_id * package_size: ((package_id + 1) * package_size)]

        for file_id in package_list:
            file_path = os.path.join(_params.dir, file_id)
            image = cv2.imread(file_path)

            if image is None:
                print("got")

            width = image.shape[1]
            height = image.shape[0]

            new_w = width + pixel * 2
            new_h = height + pixel * 2

            blank_image = np.zeros((new_h, new_w, 3), np.uint8)
            blank_image[0:new_h, 0:new_w] = (255, 255, 255)
            blank_image[pixel:new_h - pixel, pixel:new_w - pixel] = image

            package_dir = os.path.join(_params.dest_dir, str(package_id))
            if not os.path.exists(package_dir):
                os.makedirs(package_dir)
            _dest = os.path.join(package_dir, ("ext-" + file_id))
            cv2.imwrite(_dest, blank_image)

        csv_path = os.path.join(_params.dest_dir, str(package_id), "ext-{}.csv".format(package_id))
        with open(csv_path, "w") as f:
            for file_id in package_list:
                w_str = "ext-{},ext-label-{}\n".format(file_id, file_id[:-3]+"png")
                f.write(w_str)

    end = time.time()

    print("finish in {} s".format(end - start))
