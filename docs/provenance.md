# Scope, ownership and reporting decisions

This portfolio snapshot covers experiment tooling and analysis for a Vietnamese voice-to-action agent. It does not claim authorship of the pretrained ASR/LLM families, the organizers' simulator, or every team model checkpoint. The W50 adapter was supplied as a team artifact; public examples illustrate the integration and audit work around it.

## What is included

- A concise account of observed failures, fixes, tests and remaining uncertainty.
- Aggregate historical ASR/agent measurements, with input split and hardware scope.
- Newly refactored, CPU-runnable public examples of artifact identity and execution logging. These are release demonstrations, not the original full training system.
- An output-cleared runtime notebook and two small historical feature-cache modules; their lineage and adaptation are documented.

## What is intentionally absent

No raw competition dataset, audio, sample identifiers, transcripts, app observations/state, prediction rows, teacher responses, adapter/model weights, personal paths or credentials. No per-test-ID override code. The organizers' original SDK is not redistributed or replaced by the synthetic demonstration.

Working notes tracked manually edited/test-specific submissions, sometimes called “cheat” files by the project owner. The81.99 result belonged to that category. It is excluded from model-performance claims and not presented as a deployable4B system. This distinction is kept explicit rather than silently selecting the highest number for the README.

## Accountability

Development involved team-provided checkpoints and AI-assisted iteration. This repository is an edited technical record of integration, diagnosis and evaluation work; publishing it does not imply that all underlying components were developed independently by one person. Read the linked code, tests and evidence limits to assess the work.

Public/private test outputs are not provided as SFT/distillation material. Historical T1–T4 candidates were not uniformly accepted; creation counts alone do not prove label quality or distillation benefit. No new GPU experiment, private-test result, final competition ranking or production-readiness claim was added to prepare this portfolio.

The release commits are **one file per commit**, as requested. They show the public packaging sequence, not a fabricated backdated history of bug fixes. Original evidence dates remain separate from the 20 November 2026release date.
