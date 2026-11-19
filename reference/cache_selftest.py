"""Exercise the real CPU Whisper extractor and Safetensors before model download."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import safetensors
from safetensors.torch import load_file, save_file
import torch
import transformers
from transformers import WhisperFeatureExtractor

from feature_cache import packed_tensors, prepare_cached, save_features


def assert_equal(original, restored):
    assert set(original) == set(restored)
    for name, value in original.items():
        actual = restored[name]
        assert actual.device.type == 'cpu' and actual.is_contiguous(), name
        assert value.shape == actual.shape and value.dtype == actual.dtype, name
        assert torch.equal(value.detach().cpu(), actual), name


def check(directory):
    directory = Path(directory)
    cases = []
    old_failures = 0
    for feature_size in [80, 128]:
        extractor = WhisperFeatureExtractor(feature_size=feature_size)
        wave = (np.sin(np.arange(32000, dtype=np.float64) * (2 * np.pi * 220 / 16000)) * .1).astype(np.float32)
        values = dict(extractor(wave, sampling_rate=16000, return_tensors='pt', return_attention_mask=True))
        values['labels'] = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        mask = values['attention_mask']
        assert not mask.is_contiguous(), 'Expected the Whisper strided mask that failed in uploaded logs'
        try:
            save_file(values, str(directory / f'old_{feature_size}.safetensors'))
        except ValueError as error:
            assert 'non contiguous' in str(error) and 'attention_mask' in str(error), str(error)
            old_failures += 1
        else:
            raise AssertionError('Unfixed Safetensors write unexpectedly succeeded')
        fixed = directory / f'fixed_{feature_size}.safetensors'
        save_features(fixed, values)
        assert_equal(values, load_file(str(fixed)))
        calls = []
        class Task:
            def prepare(self, row):
                calls.append(row['clip_id'])
                return values
        task = Task()
        row = {'clip_id': f'cpu_synthetic_{feature_size}'}
        first = prepare_cached(task, row, directory / 'cached', 0)
        second = prepare_cached(task, row, directory / 'cached', 1)
        assert calls == [row['clip_id']], 'Cache hit must not prepare the waveform again'
        assert_equal(values, first)
        assert_equal(first, second)
        assert not mask.is_contiguous(), 'Packing must not mutate the extractor output'
        cases.append({'feature_size': feature_size, 'original_mask_stride': list(mask.stride()),
                      'mask_shape': list(mask.shape), 'round_trip_equal': True, 'cross_rank_cache_hit': True})
    # Separate entries may share source storage or hold gradient-tracking views.
    source = torch.arange(24, dtype=torch.float32).reshape(4, 6).requires_grad_()
    views = {'transpose': source.t(), 'alias': source.t(), 'slice': source[:, ::2]}
    packed = packed_tensors(views)
    assert len({v.untyped_storage().data_ptr() for v in packed.values()}) == len(packed)
    assert all(not v.requires_grad for v in packed.values())
    shared = directory / 'shared.safetensors'
    save_features(shared, views)
    assert_equal(views, load_file(str(shared)))
    # An interrupted write must retain the previous complete cache file.
    target = directory / 'atomic.safetensors'
    save_features(target, {'before': torch.arange(4)})
    before = target.read_bytes()
    import feature_cache
    writer = feature_cache.save_file
    def fail_write(payload, filename):
        Path(filename).write_bytes(b'interrupted-test-write')
        raise OSError('injected interrupted write')
    feature_cache.save_file = fail_write
    try:
        try:
            save_features(target, {'after': torch.arange(8)})
        except OSError:
            pass
        else:
            raise AssertionError('Injected failure was swallowed')
    finally:
        feature_cache.save_file = writer
    assert target.read_bytes() == before
    assert not list(directory.rglob('*.tmp'))
    return {'status': 'PASS', 'python': sys.version, 'torch': str(torch.__version__),
            'transformers': transformers.__version__, 'safetensors': safetensors.__version__,
            'platform': sys.platform, 'device': 'CPU', 'synthetic_audio_only': True,
            'old_noncontiguous_error_reproduced': old_failures == 2, 'extractor_cases': cases,
            'shared_storage_and_strided_views': 'PASS', 'atomic_failure_preserves_cache': True,
            'feature_cache_sha256': hashlib.sha256(Path(feature_cache.__file__).read_bytes()).hexdigest(),
            'not_tested': 'Unsloth GPU/DDP main training, checkpoint recovery, merge or WER'}


def main(config_path=None):
    cfg = json.loads(Path(config_path).read_text(encoding='utf-8')) if config_path else None
    parent = Path(cfg['temp_dir']) if cfg else None
    with tempfile.TemporaryDirectory(prefix='feature_cache_check_', dir=parent) as directory:
        record = check(directory)
    if cfg:
        Path(cfg['result_dir']).mkdir(parents=True, exist_ok=True)
        (Path(cfg['result_dir']) / 'feature_cache_selftest.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print('FEATURE CACHE SELFTEST', json.dumps(record, ensure_ascii=False), flush=True)
    return record


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
