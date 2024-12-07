import os
import zipfile
from typing import List, Optional
import hashlib


def compress_files(file_paths: List[str], output_zip_path: str) -> str:
    """
    Compress multiple files into a single zip file.

    Args:
    file_paths (List[str]): List of paths to files to be compressed.
    output_zip (str): Path and name of the output zip file.

    Raises:
    FileNotFoundError: If any of the input files doesn't exist.
    """
    os.makedirs(os.path.dirname(output_zip_path), exist_ok=True)
    compression = zipfile.ZIP_DEFLATED

    with zipfile.ZipFile(output_zip_path, 'w', compression=compression) as zipf:
        for file_path in file_paths:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            file_name = os.path.basename(file_path)
            zipf.write(file_path, file_name)
    return output_zip_path


def get_file_hash(file_path: str) -> str:
    """Generate the hash of a file using the specified hashing algorithm. It generates hash by content not path. """
    hash_func = hashlib.new("sha256")
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    except FileNotFoundError:
        return f"File not found: {file_path}"
    except Exception as e:
        return f"An error occurred: {str(e)}"


def find_file_by_hash(dir_path: str, hash_str: str) -> Optional[str]:
    """Get file path from the directory based on its hash"""
    if not os.path.exists(dir_path) and os.path.isdir(dir_path):
        raise ValueError(f"Directory {dir_path} does not exist")

    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

    for f in files:
        f_hash = get_file_hash(f)
        if hash_str == f_hash:
            return f
    return None


