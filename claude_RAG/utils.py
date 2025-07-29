import hashlib
import os

def get_file_hash(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def vectorstore_exists(file_hash: str) -> bool:
    return os.path.exists(f"./vectorstore/{file_hash}")
