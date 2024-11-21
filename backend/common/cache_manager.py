import time
import os
from typing import Optional

from modules.utils.paths import BACKEND_CACHE_DIR


def cleanup_old_files(cache_dir: str = BACKEND_CACHE_DIR, ttl: int = 60):
    now = time.time()
    for filename in os.listdir(cache_dir):
        filepath = os.path.join(cache_dir, filename)
        if not os.path.isfile(filepath):
            continue
        if now - os.path.getmtime(filepath) > ttl:
            os.remove(filepath)

