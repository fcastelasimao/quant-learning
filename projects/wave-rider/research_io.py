from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path


def timestamp_label() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def archive_run_outputs(source_dir: Path, target_dir: Path) -> None:
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)


def archive_selected_outputs(source_dir: Path, target_dir: Path, file_names: list[str]) -> None:
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for file_name in file_names:
        source_path = source_dir / file_name
        if source_path.exists():
            shutil.copy2(source_path, target_dir / file_name)


def save_run_metadata(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
