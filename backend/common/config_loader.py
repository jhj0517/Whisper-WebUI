from dotenv import load_dotenv
import os
from modules.utils.paths import SERVER_CONFIG_PATH, SERVER_DOTENV_PATH
from modules.utils.files_manager import load_yaml, save_yaml

import functools


@functools.lru_cache
def load_server_config(config_path: str = SERVER_CONFIG_PATH) -> dict:
    if os.getenv("TEST_ENV", "false").lower() == "true":
        server_config = load_yaml(config_path)
        server_config["whisper"]["model_size"] = "tiny"
        server_config["whisper"]["compute_type"] = "float32"
        save_yaml(server_config, config_path)

    return load_yaml(config_path)


@functools.lru_cache
def read_env(key: str, default: str = None, dotenv_path: str = SERVER_DOTENV_PATH):
    load_dotenv(dotenv_path)
    value = os.getenv(key, default)
    return value

