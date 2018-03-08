# -*-coding:utf-8-*-

import os
import time
import argparse
import cv2
import numpy as np
from process_label_local import local_road_chn_labels

if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)
    args = parser.parse_args()

    src_dir = args.dir
    dest_dir = args.dest_dir
    if not os.path.exists(src_dir):
        exit(0)

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    dir_list = os.listdir(src_dir)
    for _dir in dir_list:
        if not _dir.isdigit():
            continue

        file_list = os.listdir(os.path.join(src_dir, _dir))
        for file_id in file_list:
            if file_id.startswith("label") or file_id.startswith("ex"):
                continue
            if file_id.endswith("jpg") or file_id.endswith("txt") or file_id.endswith("csv"):
                continue

            file_path = os.path.join(src_dir, _dir, file_id)

            _dest_dir = os.path.join(dest_dir, _dir)
            if not os.path.exists(_dest_dir):
                os.makedirs(_dest_dir)
            dest_path = os.path.join(dest_dir, _dir, file_id)

            img_data = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            width = img_data.shape[1]
            height = img_data.shape[0]

            blank_image = np.zeros((height, width, 3), np.uint8)
            for label in local_road_chn_labels:
                color = label.color
                color = (color[2], color[1], color[0])
                # color = color[::-1]
                blank_image[np.where((img_data == label.id))] = color
            cv2.imwrite(dest_path, blank_image)

    end = time.time()
    print("finished in {} ms".format(str((end - start) * 1000)))