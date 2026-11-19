"""Lossless, atomic Safetensors cache for CPU Whisper features and strided masks."""
import hashlib
import os
from pathlib import Path
import tempfile

import torch
from safetensors.torch import load_file, save_file


def packed_tensors(values):
    # Whisper subsamples its mask with a stride. Safetensors requires dense,
    # contiguous tensors; clone also gives each entry independent storage.
    result = {}
    for name, value in values.items():
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
            raise TypeError(f'Feature {name} must be a dense Torch tensor')
        result[name] = value.detach().to('cpu').clone(memory_format=torch.contiguous_format)
    return result


def save_features(path, values, rank=0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = packed_tensors(values)
    fd, temporary = tempfile.mkstemp(prefix=path.stem + f'.rank{rank}.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    try:
        save_file(payload, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def prepare_cached(task, row, feature_dir, rank):
    key = hashlib.sha256(row['clip_id'].encode()).hexdigest()
    path = Path(feature_dir) / (key + '.safetensors')
    if not path.is_file():
        save_features(path, task.prepare(row), rank)
    return load_file(str(path), device='cpu')
