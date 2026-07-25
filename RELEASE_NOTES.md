# Release Notes

## Local Builder Dataset And Launch Workflow Milestone

Date: 2026-07-25

Commit: `d6b2305 Improve local builder dataset and launch workflows`

This release turns the local builder into a more beginner-friendly, safer workflow for preparing datasets, checking readiness, previewing launch commands, and running local/offline training jobs.

### Highlights

- Added first-class corpus parquet creation from dashboard and CLI flows.
- Added SFT dataset normalization, parquet export, train/val splitting, and selected Conversation Lab export.
- Added dataset validation for corpus and SFT JSONL files, including schema hints, duplicate detection, unsupported-file reporting, and active-tokenizer counts when available.
- Added launch preflight checks for tokenizer, base training/eval, chat SFT, chat RL, and chat eval.
- Added high-risk job parameter validation before launch.
- Added command preview/copy support, resume-preview command diffs, and safer pause/resume guidance.
- Added persistent dashboard job state under ignored `builder_logs/jobs.json`.
- Added failed-job diagnosis for common local setup failures such as missing `pyarrow`, missing corpus/tokenizer/checkpoints, missing paths, and OOM-like logs.
- Added runtime model listing/recommendation through the new `python -m nanochat.builder` CLI wrapper.
- Added beginner docs for `uv`, AMD Windows CPU setup, parquet support, browser tests, and Visual Studio Build Tools.

### New CLI Surface

From `nanochat-master/`:

```powershell
python -m nanochat.builder status
python -m nanochat.builder corpus validate
python -m nanochat.builder corpus convert --input notes.md --output train/notes.parquet --mode markdown_sections
python -m nanochat.builder sft validate assistant_sandbox/chat_train.jsonl
python -m nanochat.builder job preflight --job-type chat_sft --params-json "{}"
python -m nanochat.builder job preview --job-type tokenizer_train --params-json "{}"
python -m nanochat.builder runtime list-models
python -m nanochat.builder runtime recommend
```

### Optional Dependencies

Recommended beginner install from `nanochat-master/`:

```powershell
uv sync --extra cpu --extra parquet
```

On AMD Windows, `torch.cuda.is_available()` returning `False` is expected because CUDA is NVIDIA-only. This remains a valid CPU training setup.

For dashboard browser smoke tests:

```powershell
uv sync --extra browser-tests
uv run playwright install chromium
uv run python -m pytest tests/test_dashboard_playwright.py
```

### Verification

Before this release was pushed:

- `uv run python -m pytest`: 156 passed, 10 skipped
- `python -m pytest`: 144 passed, 11 skipped
- `git diff --check`: clean, with only expected Windows CRLF notices
- `uv run python -m nanochat.builder --help`: passed
- `uv run python -m nanochat.builder status`: passed

The plain Python test run showed the same existing AMD attention warnings from PyTorch; no test failures were present.

### Known Follow-Ups

- Add richer in-page approval UI for assistant actions instead of browser confirm prompts.
- Add direct "fix this first" links from readiness/preflight messages to the relevant dashboard panel.
- Add import support for alternate external chat transcript formats.
- Add near-duplicate dataset detection if exact duplicate checks are not enough.
- Add richer job-history search/export and full-log opening if job history grows large.
