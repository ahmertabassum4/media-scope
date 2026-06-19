import os
from pathlib import Path


def project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "bias-samples").is_dir() and (parent / "snapshot-samples").is_dir():
            return parent
    raise RuntimeError("Could not find the project root with bias-samples and snapshot-samples")


def activate_project_root() -> Path:
    root = project_root()
    os.chdir(root)
    return root
