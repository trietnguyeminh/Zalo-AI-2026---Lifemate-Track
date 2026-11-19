# Results with their measurement boundaries

All figures below summarize previously supplied experiment artifacts or scores reported by the project owner. Preparing this public repository did not rerun GPU training or leaderboard evaluation.

## ASR: accuracy versus time

**Input:** the locked legacy154 conversations,154 current clips plus345 history clips (**499 clips**). Each run used **2 × T4**, with conversations divided between workers. WER is lower-is-better. Timing is ASR for an entire conversation, not single-clip latency and not full-agent latency.

| Model / decoder | Current WER% | All-clip WER% | Dialogue p95, seconds |
|---|---:|---:|---:|
| PhoWhisper-large / beam 5 |9.96|9.87|33.18|
| PhoWhisper-large / greedy |10.51|10.46|25.99|
| PhoWhisper-medium / beam 5 |11.40|11.08|21.98|
| PhoWhisper-medium / greedy |11.92|11.64|18.49|
| ChunkFormer CTC / greedy |15.86|15.54|1.47|
| Wav2Vec2 / greedy |27.06|26.77|0.42|

These are final merged/reloaded decoding results, not simply the best validation-table row. The historical audit rescored9,980 hypotheses across epoch and final outputs. Whisper medium/large selected epoch 4 after 6 completed epochs. The CTC systems have different architectures/recipes; their losses are not comparable with Whisper losses.

**Decision:** large beam 5 had the lowest measured WER, but12/154 conversations already exceeded30 seconds on T4 before planning. A lower WER alone did not justify selecting it for an agent with a 30-second end-to-end A30 target. We did not convert T4 timings into predicted A30 compliance. See [CSV](../results/asr_benchmark.csv).

## Agent: measured run and owner-reported scores

| System | Public score | Evidence |
|---|---:|---|
| W50 + original Zipformer |63.50|Owner-reported; original runtime/decoder details not fully reconstructed |
| W50 + medium V2 |53.87|Owner-reported score; supplied634-record GPU output audited |

The medium V2 ASR adapter was **epoch 5/foldA,695 training sources**. It is not the full-fit epoch 4medium checkpoint in the WER table. Medium V2 also changed normalization and engine scheduling; its9.63 point regression is not proof of the isolated effect of ASR.

For the audited medium V2 run:

- 634 samples,1,933audio clips,1,489actual tool calls;630 finished,4 timeout.
- 0reported schema rejections and 0business/tool failure results. This checks execution validity, not semantic correctness.
- Mean sum of measured stages **19.60s**, p95 **37.10s**, max **47.12s** on T4. These exclude model load/warmup and are not an online A30 measurement.
- 85stage sums exceeded30 seconds; that does not establish85official scoring failures.
- The4timeouts can explain at most **0.631 score points** if all four zero-score cases became perfect. They cannot by themselves explain 9.63points.
- Largest recorded prompt8,357 tokens; largest completion137 tokens; at most8 recorded turns. Neither the 12-turn nor512-output-token cap was shown to bind in this run.

The archived parameter audit counted 4,205,751,296active language parameters; stored LLM tensors including vision/MTP totaled 4,659,865,088. Adding the medium ASR's763,857,920gave a conservative stored-element total5,423,723,008. An unused component is not removed merely because inference does not call it. These counts do not certify every other ASR combination or a final Docker package.

## Excluded from model-performance claims

The separately reported81.99 result used manual/test-specific output adjudication. It is not an automatically reproduced W50 checkpoint result, and neither those scripts nor prediction rows are included. Public/private test rows are not training/distillation labels in this release.

RescueV4, prompt-checklist and thinking-mode ablations remain **not GPU-evaluated** here. No private-test score, final ranking, production deployment or end-to-end A30 deadline result is claimed. Training provenance for the team W50 checkpoint is incomplete; training-set diagnostics are not certified unseen validation. See [provenance](provenance.md).
