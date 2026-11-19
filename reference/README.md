# Historical feature-cache repair

`feature_cache.py` and `cache_selftest.py` are copied byte-for-byte from the experiment workspace's PhoWhisper Unsloth V1.3 source. They make the non-contiguous-mask repair inspectable without publishing a whole training repository.

The self-test uses synthetic audio and a real Whisper feature extractor, not model weights or competition examples. It requires Torch, Safetensors, Transformers and NumPy. The historical recorded CPU run used Torch 2.11.0+cpu, Transformers 4.57.6 and Safetensors 0.8.0 and passed on Windows. **It was not rerun as part of the dependency-free checks in this release.**

Optional, with a compatible separately prepared environment:

```bash
python reference/cache_selftest.py
```

With no arguments the self-test uses a temporary directory. It is outside minimal CI and checks original failure reproduction, save/load equality, shared-storage independence and preservation of an old file when publication fails.

`prepare_cached` uses clip identity inside a caller-controlled feature directory. The caller must isolate that directory by feature-extraction contract. Do not treat it as a universally safe cross-model cache; the refactored `src/lifemate_portfolio/artifacts.py` demonstrates explicit contract verification separately.
