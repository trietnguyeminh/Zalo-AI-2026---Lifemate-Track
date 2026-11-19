"""Content identities and atomic publication, without loading model weights.

This public refactor demonstrates the checks used around caches/checkpoints.
It is not a complete distributed checkpoint writer or an ASR implementation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any


def fingerprint(contract: dict[str, Any]) -> str:
    """Include model, split, decoder and code identities in the caller's contract."""
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """An interrupted write must not replace the previous complete manifest.

    os.replace is atomic on the same filesystem. This does not promise directory
    fsync durability, a multi-file transaction, or a distributed writer lock.
    """
    content = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         allow_nan=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         dir=path.parent, prefix=path.name + ".",
                                         suffix=".partial", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _artifact_path(root: Path, name: str) -> Path:
    """Receipts may reference only the caller's artifact directory."""
    item = PurePosixPath(name)
    if not name or item.is_absolute() or ".." in item.parts or "\\" in name:
        raise ValueError("Artifact names must be relative POSIX paths")
    if any(p.lower().endswith("_environment") for p in item.parts) or item.name.lower() == "state.json":
        raise ValueError("App state is accessed through the SDK, not artifact receipts")
    target = (root / Path(*item.parts)).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes root")
    if target.name.lower() == "state.json" or any(p.lower().endswith("_environment") for p in target.parts):
        raise ValueError("Resolved artifact points into app state")
    return target


def verify_receipt(root: Path, receipt: dict[str, Any], contract: dict[str, Any]) -> None:
    """Reject mismatched lineage, incomplete publication, or modified files.

    SHA verification detects accidental changes; it does not authenticate an
    attacker-controlled manifest. Only accept receipts from a trusted producer.
    """
    if receipt.get("complete") is not True or receipt.get("contract_sha256") != fingerprint(contract):
        raise ValueError("Incomplete artifact or different model/data/decode contract")
    files = receipt.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("A complete receipt needs file hashes")
    for name, expected in files.items():
        if not isinstance(name, str) or not isinstance(expected, str):
            raise ValueError("Invalid receipt entry")
        path = _artifact_path(root, name)
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError("Artifact missing or changed: " + name)
