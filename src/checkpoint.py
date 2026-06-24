import json
import os


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def cached(path, fn):
    """Return cached JSON at `path` if it exists, else compute `fn()`, save, return."""
    if os.path.exists(path):
        print(f"[ckpt] load {path}")
        return load_json(path)
    obj = fn()
    save_json(obj, path)
    print(f"[ckpt] save {path}")
    return obj
