# Curated GPU runtime reference

`w50_medium_runtime_reference.ipynb` is adapted from the 16November2026W50+PhoWhisper-medium V2 notebook. Its supplied historical GPU output was the 634-record run audited in this portfolio; the owner later reported53.87. This is a regression case study, not a claim of the best configuration.

Public changes: clear output/execution metadata, retain embedded source identity, add a provenance note and make `STAGE='PREFLIGHT_ONLY'` the default. It creates no new benchmark result in this repository. Embedded code contains runtime utilities, not training rows, transcripts or model weights.

**Required external inputs, if authorized:**

| Input | Purpose |
|---|---|
| PRIVATE `nguyenminhtric/zalo-ai` | Original released metadata/audio/tools catalog/SDK; extracted canonical `train+public_test/` layout |
| Kaggle Models `nguyenminhtric/qwen3-5-4b` | Exact Qwen3.5base snapshot, verified against pinned publisher hashes |
| `nguyenminhtric/modela` | Team W50 adapter from 63.8; merge once onto the correct base |
| PRIVATE `nguyenminhtric/update-medium` | Specific medium epoch 5/foldA adapter; not the epoch 4benchmark checkpoint |

Hardware: **2 × T4**; final profile **1 × A30**. Inputs may require team access; links/handles are not a promise of public availability. Setup uses its isolated/pinned environments and may need network; workers infer from local assets offline. Private inputs cannot be recovered from this public repository.

Default preflight needs the declared inputs/hardware but does not train or run a planner. Turning on later stages requires the original competition permissions and all assets. The historical development budget45s is not the final30sA30requirement. The notebook is not a complete offline Docker submission or a new validated GPU run.
