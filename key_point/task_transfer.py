# -*-coding:utf-8-*-

import os
import time
import json
import argparse
import shutil


if __name__ == '__main__':
    time1 = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    args = parser.parse_args()

    src_dir = args.dir
    if not os.path.exists(src_dir):
        print("img_dir[{}] is not exist!\n".format(src_dir))
        exit(1)

    img_list = []
    dir_content = os.listdir(src_dir)
    for id_ in dir_content:
        if not id_.isdigit():
            continue

        csv_file_name = str(id_) + ".csv"
        csv_file = os.path.join(src_dir, str(id_), csv_file_name)

        with open(csv_file, "w") as f:
            file_list = os.listdir(os.path.join(src_dir, str(id_)))

            for file_id in file_list:
                if not file_id.endswith("jpg"):
                    continue

                cpy_file = os.path.join(src_dir, str(id_), "label-"+file_id)
                src_path = os.path.join(src_dir, str(id_), file_id)
                shutil.copy(src_path, cpy_file)

                _str = "{},label-{}\n".format(file_id, file_id)
                f.write(_str)

    print ("finish")
