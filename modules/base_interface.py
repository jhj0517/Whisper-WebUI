import os
import torch
from typing import List


class BaseInterface:
    def __init__(self):
        pass

    @staticmethod
    def release_cuda_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()

    @staticmethod
    def remove_input_files(file_paths: List[str]):
        for file_path in file_paths:
            if not os.path.exists(file_path):
                continue
            os.remove(file_path)
