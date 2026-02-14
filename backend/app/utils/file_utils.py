import os
import shutil
from typing import Optional


def ensure_dir(path: str) -> None:
    """Ensure a directory exists."""
    os.makedirs(path, exist_ok=True)


def safe_delete(path: str) -> bool:
    """Safely delete a file or directory."""
    try:
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
        return True
    except Exception:
        return False


def get_file_size(path: str) -> Optional[int]:
    """Get file size in bytes."""
    try:
        return os.path.getsize(path)
    except Exception:
        return None


def get_dir_size(path: str) -> int:
    """Get total size of a directory."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return total


def copy_file(src: str, dst: str) -> bool:
    """Copy a file."""
    try:
        ensure_dir(os.path.dirname(dst))
        shutil.copy2(src, dst)
        return True
    except Exception:
        return False
