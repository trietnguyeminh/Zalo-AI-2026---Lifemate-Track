# Architecture and execution contracts

**Input:** current Vietnamese audio, earlier user turns, supplied context/history and installed tool schemas. App observations are obtained exclusively through the original organizer SDK. Development uses **2 × T4**; final target is **1 × A30**.

## Stages

1. **Identify inputs.** Pin the model revision, adapter SHA, decoding settings, source version and evaluation split. Reject ambiguous or incompatible mounts before expensive setup.
2. **Transcribe.** Process the current request and user audio in history. Cache only under a matching identity contract; keep transcript latency separate from planner latency.
3. **Plan and observe.** Render context, installed tool schemas and supplied prior observations. Qwen proposes tools; validation precedes execution. Feed real tool results into the next model turn.
4. **Record effects.** Create one original ToolEnv per sample/job, retain every actual invocation and result, and obtain the final hash from its public SDK interface.
5. **Publish artifacts.** Validate coverage and call-log agreement before exporting. A partial run or continuation checkpoint must not be presented as a completed submission.

The user history already represents earlier effects; it is not replayed. A successful missing-information call ends the current turn after any independent work already completed. Unsupported requests may require no app calls.

## Why separate ASR and planner workers?

The medium V2 experiment ran ASR on two workers, released those GPU processes, then ran two planner shards. This avoids keeping both model families resident in each T4. It also allows reuse of a verified ASR cache while comparing a planner policy. Summing the measured stages is useful for diagnosis but is not an online A30 service measurement.

## Component lineage

| Component | Role | Public release |
|---|---|---|
| Zipformer F1 | ASR for the W50 control | Architecture/identity notes; no weights or claim that its training lineage was audited |
| PhoWhisper medium/large | ASR fine-tuning and speed/accuracy experiments | Aggregate results; one runtime reference |
| Qwen3.5-4B + W50 | Tool-use planner, team-provided LoRA | Model/adapter contract; no weights |
| Qwen3.8 language-only branch | Separate training/resume experiments | Debugging history, not interchangeable weights |
| Original `zaic_tools` | Competition app execution and state hash | Not redistributed |

The [configuration example](../configs/runtime.example.json) describes the W50 control. It is documentation, not a complete deployment configuration.

## Executable public slice

`artifacts.py` demonstrates contract fingerprints, same-filesystem atomic JSON publication and file verification. `ledger.py` accepts already-selected proposals and a factory providing the environment; it is intentionally not a planner. `synthetic_demo.py` supplies a small invented environment to make these contracts inspectable without private data.

There is no claim of distributed exactly-once execution: a crash between a real side effect and durable logging is a separate production concern. The examples preserve diagnostics and prohibit automatic retry after a tool exception. A production service would need stronger transaction/idempotency guarantees without violating the competition's call-log contract.
