import os
import numpy as np

def get_data_dir(data_type):
    # __file__ is Storage.py's own path; abspath+dirname anchors to its actual
    # location on disk (src/), regardless of what directory the caller is run from.
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)          # one level up from src/ -> braingpt-mini/
    data_dir = os.path.join(project_root, "data", data_type)
    os.makedirs(data_dir, exist_ok=True)
    return data_dir

def store_data(subjects_dict):
    for sub_id, sub_data in subjects_dict.items():
        data_type = sub_data["data_type"]
        dir_path = get_data_dir(data_type)
        file_path = os.path.join(dir_path, sub_data["tag"], f"{sub_id}.npz")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.savez_compressed(file_path, **sub_data)
        print(f"subject {sub_id} stored at {file_path}")
