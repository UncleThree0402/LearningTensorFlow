import os
import pathlib
import numpy as np


def check_file(data_dir):
    for dirpath, dirnames, filenames in os.walk(data_dir):
        print(f"There are {len(dirnames)}, {len(filenames)} files in {dirpath}")


def get_classes_name(train_dir):
    data_dir = pathlib.Path(train_dir)
    class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))
    print(class_names)
    return class_names
