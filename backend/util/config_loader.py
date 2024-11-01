from modules.utils.paths import SERVER_CONFIG_PATH
from modules.utils.files_manager import load_yaml


def load_server_config(config_path: str = SERVER_CONFIG_PATH) -> dict:
    return load_yaml(config_path)


