import time
import os
from typing import Optional

from modules.utils.paths import BACKEND_CACHE_DIR


def cleanup_old_files(cache_dir: str = BACKEND_CACHE_DIR, ttl: int = 60):
    now = time.time()
    place_holder_name = "cached_files_are_generated_here"
    for root, dirs, files in os.walk(cache_dir):
        for filename in files:
            if filename == place_holder_name:
                continue
            filepath = os.path.join(root, filename)
            if now - os.path.getmtime(filepath) > ttl:
                try:
                    os.remove(filepath)
                except Exception as e:
                    print(f"Error removing {filepath}")
                    raise
