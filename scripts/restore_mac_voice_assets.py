"""Verify and restore downloaded private voice-model backup parts outside Git."""

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def member(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError("Backup path escapes its directory")
    return path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def restore(source, output):
    manifest = json.loads((source / "backup_manifest.json").read_text())
    archive = member(source, manifest["metadata_archive"])
    if sha256(archive) != manifest["metadata_sha256"]:
        raise ValueError("Private metadata archive checksum mismatch")
    target = member(output, "moss-checkpoint/model.safetensors")
    if target.exists():
        raise ValueError("Destination already has a model; choose an empty directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".partial")
    combined = hashlib.sha256()
    with temporary.open("wb") as dst:
        for part in manifest["parts"]:
            digest = hashlib.sha256()
            with ExitStack() as stack:
                if "archive_name" in part:
                    path = member(source, part["archive_name"])
                    if sha256(path) != part["archive_sha256"]:
                        raise ValueError("Model part archive checksum mismatch")
                    packed = stack.enter_context(zipfile.ZipFile(path))
                    if packed.getinfo(part["name"]).file_size != part["bytes"]:
                        raise ValueError("Model part size mismatch")
                    src = stack.enter_context(packed.open(part["name"]))
                else:
                    path = member(source, part["name"])
                    if path.stat().st_size != part["bytes"]:
                        raise ValueError("Model part size mismatch")
                    src = stack.enter_context(path.open("rb"))
                while block := src.read(8 * 1024 * 1024):
                    digest.update(block)
                    combined.update(block)
                    dst.write(block)
            if digest.hexdigest() != part["sha256"]:
                raise ValueError("Model part checksum mismatch")
    if combined.hexdigest() != manifest["checkpoint_sha256"]:
        raise ValueError("Restored model checksum mismatch")
    with zipfile.ZipFile(archive) as bundle:
        paths = [(info, member(output, info.filename)) for info in bundle.infolist()]
        for info, path in paths:
            if path == target or path == temporary:
                raise ValueError("Metadata archive must not replace model weights")
            if path.exists():
                raise ValueError("Refusing to overwrite existing private assets")
        for info, path in paths:
            if info.is_dir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(info) as src, path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    temporary.replace(target)
    shutil.copy2(source / "backup_manifest.json", output / "backup_manifest.json")
    print("Restored verified voice assets:", output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--downloads", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    restore(args.downloads, args.output)


if __name__ == "__main__":
    main()
