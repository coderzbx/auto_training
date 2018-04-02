import os
import shutil

for i in range(2580, 2805):
    dir_path = os.path.join(".", str(i))
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)