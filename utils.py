import json
import os
import os.path as osp
import random
import shutil
import time
import zipfile
from typing import Any, Union, Optional

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from huggingface_hub import hf_hub_download
from transformers import set_seed
import re


def extract_tag_content(text: str, tag: str):
    pattern = fr"<{tag}>(.*?)</{tag}>"
    return re.findall(pattern, text)


def save_json(file_path: str, data: Any, mode="w"):
    with open(file_path, mode, encoding='utf-8') as f:
        json.dump(data, f)


def save_list_json(file_path: str, data: Any):
    with open(file_path, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")


def save_entry_to_list_json(file_path: str, data: Any):
    with open(file_path, "a") as f:
        f.write(json.dumps(data) + "\n")


def load_json(file_path: str) -> Union[dict, list[dict]]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_list_json(file_path: str) -> list[dict]:
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data


def make_dir(dir_name: str):
    os.makedirs(dir_name, exist_ok=True)


def is_main_process():
    try:
        # Hugging Face Accelerate
        accelerator = Accelerator()
        return accelerator.is_main_process
    except ImportError:
        pass  # Accelerate not installed

    try:
        # PyTorch Distributed
        return not dist.is_initialized() or dist.get_rank() == 0
    except ImportError:
        pass  # PyTorch not installed

    # Default to True for single-process execution
    return True


def download_hf_file(repo_id,
                     filename,
                     local_dir,
                     subfolder=None,
                     repo_type="dataset",
                     cache_dir=None,
                     ) -> str:
    hf_hub_download(repo_id=repo_id, subfolder=subfolder, filename=filename, repo_type=repo_type,
                    local_dir=local_dir, local_dir_use_symlinks=False, cache_dir=cache_dir, force_download=True)
    if subfolder is not None:
        shutil.move(osp.join(local_dir, subfolder, filename), osp.join(local_dir, filename))
        shutil.rmtree(osp.join(local_dir, subfolder))
    return osp.join(local_dir, filename)


def extract_zip(path: str, folder: str):
    r"""Extracts a zip archive to a specific folder.
    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(folder)


def gen_random_seed() -> int:
    return int(time.time() * 1000) % (2 ** 32 - 1)


def set_random_seed(seed: Optional[int] = None) -> int:
    if seed is None:
        seed = gen_random_seed()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    print(f"Random seed is set to {seed}...")
    return seed


class Timer:
    def __init__(self):
        self.total_time = 0.0
        self._start_time = None

    def start(self):
        if self._start_time is None:
            self._start_time = time.time()
        else:
            raise RuntimeError("Timer is already running. Call end() before calling start() again.")

    def end(self):
        if self._start_time is not None:
            elapsed = time.time() - self._start_time
            self.total_time += elapsed
            self._start_time = None
        else:
            raise RuntimeError("Timer is not running. Call start() before calling end().")

    def return_time(self):
        current_total = self.total_time
        if self._start_time is not None:
            # Include time since last start
            current_total += time.time() - self._start_time
        return current_total

    def reset(self):
        self.total_time = 0.0
        self._start_time = None