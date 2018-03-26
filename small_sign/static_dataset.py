# -*-coding:utf-8-*-

import os
import shutil
import time
from PIL import Image


if __name__ == '__main__':
    dir_path = "/data/deeplearning/dataset/small_sign"
    dest_path = "/data/deeplearning/dataset/small_sign/filter"
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    dir_list = os.listdir(dir_path)
    dir_list.sort()

    total_count = 0
    target_count = 0

    for dir_name in dir_list:
        start_ = time.time()
        total_ = 0
        target_ = 0
        dir_ = os.path.join(dir_path, dir_name)

        for dir_id in ["instructive", "prohibition", "warning"]:
            target_dir = os.path.join(dir_, "labels", dir_id)

            if not os.path.exists(target_dir):
                continue

            file_list = os.listdir(target_dir)
            for file_id in file_list:
                if not file_id.endswith("jpg"):
                    continue

                total_count += 1
                total_ += 1
                image = Image.open(os.path.join(target_dir, file_id))
                w, h = image.size

                if w >= 100 and h >= 100:
                    target_count += 1
                    target_ += 1
                    dest_dir = os.path.join(dest_path, dir_id)
                    if not os.path.exists(dest_dir):
                        os.makedirs(dest_dir)
                    shutil.copy(os.path.join(target_dir, file_id), os.path.join(dest_dir, file_id))
        end_ = time.time()
        print("{} fliter {}->{} in {}s".format(
            dir_name, total_, target_, (end_ - start_)
        ))
    print("finish {}->{}".format(total_count, target_count))