import os
import pathlib
import numpy as np


def check_file(data_dir):
    for dirpath, dirnames, filenames in os.walk(data_dir):
        print(f"There are {len(dirnames)}, {len(filenames)} files in {dirpath}")

