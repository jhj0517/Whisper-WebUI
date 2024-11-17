import os
import zipfile
from typing import List


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
