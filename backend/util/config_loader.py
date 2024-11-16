from dotenv import load_dotenv
import os
from modules.utils.paths import SERVER_CONFIG_PATH, SERVER_DOTENV_PATH
from modules.utils.files_manager import load_yaml

import functools


@functools.lru_cache
def load_server_config(config_path: str = SERVER_CONFIG_PATH) -> dict:
    return load_yaml(config_path)


@functools.lru_cache
def read_env(key: str, dotenv_path: str = SERVER_DOTENV_PATH):
    load_dotenv(dotenv_path)
    value = os.getenv(key)

    if value is None:
        raise KeyError(f"Key {key} does not exist in dotenv")

    return value

