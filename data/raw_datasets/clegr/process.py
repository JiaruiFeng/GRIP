import json
from os import path as osp

import pandas as pd
import torch
from tqdm import tqdm

from utils import make_dir, download_hf_file, extract_zip, save_list_json

HF_REPO_ID = "tenseisoham/CLEGR"


def process_clegr(save_dir: str):
    make_dir(save_dir)
    download_hf_file(HF_REPO_ID, subfolder="data", filename="data_for_hf.tar.gz", local_dir=save_dir)
    extract_zip(osp.join(save_dir, "data_for_hf.tar.gz"), save_dir)


if __name__ == "__main__":
    process_clegr("outputs/data/clegr")