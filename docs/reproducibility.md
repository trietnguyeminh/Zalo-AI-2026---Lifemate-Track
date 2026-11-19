# Reproducibility: runnable checks and restricted experiments

## Public CPU path

**Input:** invented bytes, dictionaries and a scripted synthetic tool plan. Python 3.11+, no external packages or network:

```bash
python tools/run_checks.py
python examples/synthetic_demo.py
```

The current standard-library suite has16 tests. They verify artifact mismatch/failure behavior and the execution-ledger contract. They do not prove CUDA, NCCL, ASR quality or LLM tool-use quality. CI runs the same commands on Python 3.11/3.12; its badge reports actual workflow status.

The optional historical feature-cache self-test requires Torch, Safetensors, Transformers and NumPy, but uses synthetic audio and no model weights. It is outside the minimal CI dependency set. Historical validation results are marked as such; see [reference/README.md](../reference/README.md).

## Reference GPU notebook

See [notebooks/README.md](../notebooks/README.md). Input is authorized competition metadata/audio/SDK, the exact public Qwen base, the team W50 adapter and the specific ASR checkpoint. Default development hardware is **2 × T4**; final target **1 × A30**. Setup may use Internet for pinned dependencies/base metadata; inference workers use local assets offline.

The public copy is output-cleared and defaults to **PREFLIGHT_ONLY**. It is an adapted reference from the audited medium V2 lineage, not a newly measured or guaranteed working Kaggle submission. Restricted assets are not supplied by this repository.

## Run identity and checkpoints

Record base revision, adapter hashes, tokenizer/config hashes, split identity, decode settings, source hashes, runtime versions, hardware, seed and actual outputs. Keep greedy decoding and fixed seeds; do not assume seeds alone guarantee bitwise identity across kernels/engines/hardware.

Checkpoint completeness means more than an adapter file: optimizer, scheduler, scaler, RNG state and successful update count must be consistent. Use a completion receipt and reject mismatched fingerprints. A checkpoint at 92 updates is not completion of a 174-update run. Do not merge/export or run final evaluation merely because the time budget ended.

## Split discipline

Split conversations before flattening current/history clips. Legacy ASR comparisons use a fixed1,391 fit/154 dev partition; keep their499evaluation clips intact. An existing ASR checkpoint had seen114 samples in a newer agent-system dev split. It must be benchmarked on its legacy split or retrained on the locked system fit before claiming a clean system-dev result.

The W50 training manifest is not complete enough to certify its dev set as unseen. Dev remains diagnostic until provenance is reconciled. Do not reshuffle the evaluation set to remove inconvenient cases, select checkpoints with public-test labels, or import unreviewed synthetic candidates as accepted training data.

## Evidence hierarchy

1. Supplied GPU artifact/run log: establishes what the run actually produced.
2. CPU reproduction/parity test: establishes a local property under that setup.
3. Owner-reported leaderboard score: recorded accurately, not independently portal-verified.
4. Proposed repair/ablation: a hypothesis until the intended hardware experiment runs.

Hash manifests detect accidental mismatches; they do not independently authenticate an untrusted producer. Original logs and row-level results stay outside this public repository. [Aggregate evidence](../results/evidence_summary.json) identifies the archived sources without reproducing their contents.
