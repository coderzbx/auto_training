# -*-coding:utf-8-*-

import os
import time
import json
import argparse
import shutil


if __name__ == '__main__':
    time1 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)
    parser.add_argument('--start', type=int, required=True, default=1)
    parser.add_argument('--step', type=int, required=False, default=20)
    args = parser.parse_args()

    src_dir = args.src_dir
    if not os.path.exists(src_dir):
        print("img_dir[{}] is not exist!\n".format(src_dir))
        exit(1)
    dest_dir = args.dest_dir
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

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

    print ("\nstart to divide\n")
    step = 20
    if args.step:
        step = args.step

    types = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"]

    # 数据包从1开始
    total_index = 0
    package_index = 425
    package_annotations = {"types": types}
    package_imgs = {}

    # step个数图片一个包
    image_index = 0

    package_dir = os.path.join(dest_dir, str(package_index))
    if not os.path.exists(package_dir):
        os.makedirs(package_dir)
    for id_ in img_list:
        image_index += 1

        path = id_
        src_image = os.path.join(src_dir, path)

        path_list = path.split(".")
        image_name = path_list[0]

        dest_image = os.path.join(package_dir, id_)
        shutil.copy(src_image, dest_image)

        package_imgs[image_name] = {"path": path, "objects": []}

        if image_index == step:
            package_annotations["imgs"] = package_imgs
            annot_name = str(package_index) + ".json"
            annot_file = os.path.join(package_dir, annot_name)

            with open(annot_file, "w") as f:
                json.dump(package_annotations, f)

            package_index += 1
            image_index = 0
            package_imgs = {}
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

    print ("finish")
