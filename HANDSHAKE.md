# LLM Tweaker Builder - Handshake

## Module identity (required)

- **module_id**: `llm_tweaker_builder`
- **display_name**: `LLM Tweaker Builder`

## What this module is for (required)

LLM Tweaker Builder is a local-only dashboard for training and testing your own nanochat-based models. It keeps the full builder workflow in one place: local corpus setup, tokenizer training, base-model training, chat SFT, local runtime assistance, and checkpoint testing.

## Inputs this module expects (required)

- Local corpus files in `nanochat-master/local_corpus/`
- Local GGUF helper models in `nanochat-master/Assistant_models/`
- Design and identity choices entered in the dashboard
- Optional draft chat datasets and notes in `nanochat-master/Assistant_sandbox/`
- Local Python environment with required packages already installed

## Outputs this module produces (required)

- Tokenizer artifacts created by the builder pipeline
- Base-model, SFT, and RL checkpoints created by local jobs
- Local logs in `nanochat-master/builder_logs/`
- Draft and published training data in `nanochat-master/Assistant_sandbox/`
- A working browser dashboard for launching, monitoring, and testing the builder workflow

## Suspend rundown template (required)

> **Status:** Builder stage, latest job status, and current testing state.
> **What changed:** Corpus/design edits, jobs launched or completed, and any new checkpoints or datasets created.
> **Open questions:** Missing files, package issues, hardware limits, or unclear next training decisions.
> **Next action:** The next concrete builder step such as draft data, train tokenizer, run base training, run SFT, or test a checkpoint.
> **Artifacts:** `nanochat-master/builder_logs/activity.jsonl`, `nanochat-master/builder_logs/*`, `nanochat-master/Assistant_sandbox/*`, `nanochat-master/local_corpus/*`

## Portable bridge note

This wrapper folder exists so ChattyCog can host the original builder dashboard that lives in `nanochat-master/` without changing the app itself.
