# From failed runs to testable fixes

Inputs: supplied Kaggle notebooks/logs, result archives, and the corresponding source snapshots. GPU experiments used **2 × T4**; the final hardware target is **1 × A30**. No competition audio, request rows or app observations are reproduced here.

The structure of each incident is intentional: observation → evidence → change → check → remaining uncertainty. A plausible explanation is not a confirmed root cause, and a CPU regression test is not a successful GPU training run.

## 1. NumPy import failure before training

**Observation.** A Python 3.13 environment failed while importing Torch through NumPy, before model loading. The outer error suggested importing NumPy from its source directory.

**Evidence.** The traceback pointed to NumPy 1.26.4 under a generic distribution site path. The bootstrap had sorted host paths and placed an ABI-incompatible package ahead of the intended environment. Looking only at the final exception would have led to the wrong repair.

**Change.** Create an isolated environment, retain only ABI-compatible host paths, install a compatible binary NumPy wheel before importing Torch, and remove inherited `PYTHONPATH`/`PYTHONHOME` from children. Keep the notebook's GPU stack intact.

**Check.** The historical CPU reproduction inserted an incompatible Python 3.12 wheel into a Python 3.13 path, reproduced the extension failure, then verified the repaired import and a NumPy operation. The recorded regression suite passed 14 CPU tests and skipped1 GPU test.

**Limit.** That reproduction ran on Windows. It did not certify Linux/CUDA training. Later logs were needed to confirm the next stage was reached. [NumPy troubleshooting](https://numpy.org/doc/stable/user/troubleshooting-importerror.html) supports checking interpreter, NumPy version and imported paths rather than treating every import failure as the same issue.

## 2. Safetensors rejected the attention-mask cache

**Observation.** Setup and a two-update GPU preflight passed, but the main run failed while preparing epoch-zero evaluation. No main-training checkpoint or new WER existed yet.

**Evidence.** Whisper's attention mask was a strided view; the recorded stride was `[480000,160]`. The feature-cache writer passed it directly to Safetensors. The NCCL cleanup warning occurred after the actual exception.

**Change.** Detach tensors, move them to CPU, clone contiguous storage for each entry, write to a process-specific temporary file, and publish with `os.replace`. Do not change tensor values or the training recipe to fix serialization.

**Check.** The historical CPU self-test used a real Whisper feature extractor with **synthetic audio**, both80/128mel bins, strided/shared-storage views, save/load equality and injected write failure. It reproduced the old exception and preserved the previous complete cache when publication failed. The actual small modules are preserved in [reference/](../reference/README.md). Safetensors documents the [dense/contiguous requirement](https://huggingface.co/docs/safetensors/api/torch#safetensorstorchsave_file).

**Limit.** Current minimal tests do not import Torch. The historical GPU runs and later ASR outputs are reported separately. The historical feature key uses clip identity within a caller-isolated directory; it is not a universal cross-experiment cache key.

## 3. OOM despite a successful longest-sequence probe

**Observation.** Training requested another3.21 GiB on one T4. DDP had a model replica on each GPU: two16GB cards did not provide one32GB model-memory pool. A later longest-sequence forward/backward probe passed at roughly 8.12 GiB peak, but actual training still ran out of memory.

**Evidence.** The first probe did not reproduce all optimizer, gradient-scaling, accumulation and padding conditions. Source inspection also showed different GQA/SDPA paths depending on mask shape. A quadratic attention allocation was a plausible contributor, but logs lacked a kernel trace proving it.

**Change.** Add saved-tensor CPU offload, smaller loss chunks and DDP bucket views. Make the training probe closer to the real scaled accumulation window. In a separate revision, guard attention dispatch and fail clearly if an efficient kernel is unavailable rather than silently falling back. Preserve valid tokens and labels instead of shortening the dataset to manufacture success.

**Check.** Historical CPU tests compared loss/gradient parity for offload and explicit repeated-KV attention. The supplied follow-up run reached21/174 updates before an AMP-related abort.

**Limit.** The tests did not measure CUDA transfer cost or establish the suspected kernel as the root cause. The later training progress is evidence of progress, not proof that every memory issue was solved.

## 4. Overflow was treated as a fatal event; session time was limited

**Observation.** A revision aborted when GradScaler reduced its scale. A larger model run elsewhere also exceeded a 12-hour session. Restarting from scratch wasted completed work.

**Change.** Coordinate nonfinite detection across ranks; back off the scale and replay the entire accumulation window with its RNG state. Advance optimizer/scheduler/update counters only after a successful step. Save adapter, optimizer, scheduler, scaler, RNG and progress; gate merge/evaluation/submission on training completion. On budget exhaustion, export a continuation package and report `RESUME_REQUIRED`.

**Check.** Historical CPU tests injected nonfinite gradients and checked replay/resume parity. A subsequently supplied artifact contained a real checkpoint at **update92**, including optimizer/scheduler progress and per-rank RNG state. That is stronger evidence than a log saying an archive should have been written.

**Limit.** The92-update checkpoint was not174/174 completion. Some local distributed tests could not run because Windows sockets were restricted; those tests were not reported as passed. Changing the AMP policy also prevents a claim of bitwise equivalence to the old runner.

## 5. Model mounts, adapter lineage and stale caches

**Observation.** An input model was declared in Kaggle metadata but the resolver could not find it at a hardcoded path. Similar names also made it possible to confuse a public base, a language-only export, an adapter and an already-merged model.

**Change.** Inventory permitted mount locations before downloads, validate config/shards, require an explicit choice for ambiguous matches, and fail rather than silently substitute another checkpoint. For W50, verify base files against the pinned publisher revision before a single LoRA merge. Bind decoded caches to audio, model, decoder, split, code and runtime identities.

**Check.** The recorded resolver suite passed 19fixture regressions, including aliases, nested paths, ambiguity, missing shards and protected state trees. The W50 V2 checks later passed 38 CPU cases with 1platform-dependent skip; the supplied GPU output separately established a 634-sample run. The public [artifact utility](../src/lifemate_portfolio/artifacts.py) tests the distilled identity/publication contract with invented files.

**Limit.** A model handle is not a weight checksum. A toy shard does not establish that real weights load. Do not resume across a changed model/data/recipe fingerprint or claim a clean validation split without training provenance.

## 6. A telemetry flag incorrectly implied truncation

**Observation.** Every returned Whisper sequence in an audited run lacked an EOS token according to a diagnostic flag. Taking that literally would have labeled all clips truncated.

**Evidence and correction.** The inspected Transformers 4.57.6 Whisper generation path removes terminal EOS during postprocessing. The diagnostic examined already-processed output, so the flag could not distinguish valid endings from budget exhaustion. The report withdrew that interpretation rather than expanding the generation budget on a false premise.

**Next check.** Record actual generation length/finish evidence at the correct stage. The confirmed truncation count remained unknown. [Inspected generation code](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/whisper/generation_whisper.py).

## 7. Valid tool calls were still not enough

**Observation.** W50+medium V2 had no reported schema rejections or business/tool failures, yet the owner-reported score was53.87. That did not make its actions semantically correct. Earlier requests for bigger step/token limits also needed measurement: this run used at most8 recorded generation turns and 137 output tokens/turn, below the 12/512 caps.

**Decision.** Separate transcript quality, argument grounding, planning, stop conditions and latency. Freeze weights/ASR/runtime while testing prompt changes on training diagnostics. Compare oracle text with audio input before blaming ASR alone. Treat more tool calls as a possible F1/state risk, not evidence of more useful reasoning.

**Status.** Prompt/checklist,20/1024budget and thinking-mode ablations are proposals, not measured improvements in this release. Manual/test-specific output edits were tracked separately and excluded from model results.
