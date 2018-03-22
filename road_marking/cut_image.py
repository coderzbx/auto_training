import cv2
import os

image_dir = "/data/deeplearning/dataset/training/data/local/fisheye"
dir_list = os.listdir(image_dir)
for dir_ in dir_list:
    if not str(dir_).isdigit():
        continue

    dir_path = os.path.join(image_dir, dir_)
    file_list = os.listdir(dir_path)

    for file_id in file_list:
        if not file_id.endswith("jpg") and not file_id.endswith("png"):
            continue
        file_path = os.path.join(dir_path, file_id)
        image = cv2.imread(file_path)

        width = image.shape[1]
        height = image.shape[0]

        new_image = image[50:height-50, 50:width-50]
        new_path = os.path.join(dir_path, file_id[4:])
        cv2.imwrite(new_path, new_image)