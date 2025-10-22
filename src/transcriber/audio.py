from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
import shutil
from typing import List, Tuple, Optional

AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}


def is_audio_file(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_EXTS


def _extract_zip(zip_path: Path) -> Tuple[List[str], str]:
    """Safely extract a ZIP to a temporary directory and return (files, tmp_root).

    Only supported audio files are extracted. Path traversal ("zip slip") is prevented by
    validating each member path stays within the temporary root. Symlinks are not created.
    """
    tmp_root = Path(tempfile.mkdtemp(prefix="transcriber_zip_"))
    root_abs = tmp_root.resolve()
    audio_paths: List[str] = []

    def _is_within_root(target: Path) -> bool:
        try:
            target_abs = target.resolve()
        except FileNotFoundError:
            # Parent may not exist yet; resolve non-strictly
            target_abs = (target.parent.resolve() / target.name).resolve()
        root_str = str(root_abs) + os.sep
        return str(target_abs).startswith(root_str)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            # Normalize to POSIX inside-zip paths and strip leading slashes
            rel = PurePosixPath(info.filename.lstrip("/\\"))
            if rel.name == "":
                # Directory entry
                continue
            # Skip non-audio files early based on suffix
            if rel.suffix.lower() not in AUDIO_EXTS:
                continue
            dest = tmp_root.joinpath(*rel.parts)
            # Prevent path traversal
            if not _is_within_root(dest):
                # Clean up the temp dir to avoid leaking partial extractions
                shutil.rmtree(tmp_root, ignore_errors=True)
                raise ValueError(f"Unsafe path in ZIP: {info.filename}")

            dest.parent.mkdir(parents=True, exist_ok=True)
            # Extract file content safely (avoid creating symlinks)
            with zf.open(info, "r") as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
            audio_paths.append(str(dest))

    audio_paths.sort()
    return audio_paths, str(tmp_root)


def gather_inputs(path: str) -> Tuple[List[str], Optional[str]]:
    """Return (audio_files, tmp_root) from a file, directory, or ZIP archive.

    Caller is responsible for cleaning up tmp_root if provided.
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(target)

    if target.is_file():
        if target.suffix.lower() == ".zip":
            return _extract_zip(target)
        return ([str(target)] if is_audio_file(target) else []), None

    return (
        sorted(
            str(f)
            for f in target.rglob("*")
            if f.is_file() and is_audio_file(f)
        ),
        None,
    )

def cleanup_tmp(root: Optional[str]) -> None:
    if root:
        try:
            shutil.rmtree(root)
        except Exception:
            pass
