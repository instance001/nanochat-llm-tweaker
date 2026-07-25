# LLM Tweaker Builder Build Plan

This is the working build plan for improving the local builder. It is ordered by biggest practical payoff first: features that reduce user confusion, prevent wasted training runs, and make dataset creation more first-class.

## Guiding Priorities

1. Make dataset creation and validation trustworthy before training starts.
2. Keep local/offline behavior as a core invariant.
3. Close CLI/GUI gaps where the same capability already exists in one surface.
4. Add preflight checks before expensive or long-running actions.
5. Prefer small, testable backend helpers before large dashboard UI changes.

## Phase 1: First-Class Dataset Creation

### 1. Parquet Dataset Creation

Status: initial implementation added.

**Goal:** Let users create valid `.parquet` corpus files from the dashboard without hand-crafting structured records.

Current state:

- `CorpusManager.write_parquet_file()` can write parquet from records.
- `/api/corpus/write` can accept `records` for `.parquet`.
- The dashboard editor can save parquet if the user provides JSON object rows or an array.
- The dashboard now has a Create Parquet Corpus flow that can preview/write rows from paragraphs, lines, markdown sections, JSONL objects, or a JSON object/array.

Build:

- Done: add a backend helper that converts text input into record rows.
- Done: support conversion modes:
  - one row per paragraph
  - one row per line
  - one row per markdown heading section
  - JSONL objects
  - JSON array of objects
- Done: add preview output before writing:
  - row count
  - columns
  - sample rows
  - rejected/empty rows
- Done: add a dashboard control near Corpus Editor:
  - target path
  - conversion mode
  - text input or current editor content
  - preview
  - write parquet

Suggested backend files:

- `nanochat/sandbox_tools.py`
- `scripts/chat_web.py`

Suggested tests:

- Done: text paragraphs to parquet records
- Done: JSONL/JSON array conversion coverage
- Done: markdown sections to parquet records
- Done: invalid JSONL reports useful errors
- Done: converted parquet write path
- Still useful later: append-mode UI coverage

### 2. SFT Dataset Export Options

Status: initial implementation added.

**Goal:** Keep JSONL as the canonical SFT format, but allow export/conversion when users want structured dataset files.

Current state:

- SFT drafting writes conversation JSONL in `assistant_sandbox`.
- `sft_dataset_tools.py` validates/normalizes conversations.
- Corpus parquet support is not the same thing as SFT parquet support.
- The Sandbox Editor exposes explicit normalized JSONL and parquet export actions.
- `/api/sandbox/sft/export` supports preview, canonical JSONL, and parquet exports.

Build:

- Done: explicit "Export SFT Parquet" action.
- Done: explicit "Normalize SFT JSONL" action.
- Done: explicit "Split SFT Train/Val" action.
- Done: supported exports:
  - canonical JSONL
  - parquet with one conversation per row
  - parquet with columns like `messages`, `turn_count`, `source`
- Done: deterministic JSONL train/val split export with ratio and seed controls.
- Done: preview action for JSONL, parquet, and split exports before writing files.
- Done: direct parquet preview/loading in the Sandbox Editor.
- Done: SFT exports stay scoped to `assistant_sandbox` and are clearly separate from corpus parquet creation.

Suggested backend files:

- Done: `nanochat/sft_dataset_tools.py`
- Done: `scripts/chat_web.py`
- Done: `nanochat/dashboard.html`

Suggested tests:

- Done: valid conversation JSONL to parquet records
- Done: invalid role/order rejected by existing SFT validation coverage
- Done: exported parquet preserves message structure through the dashboard API

Still useful later:

- Add transcript import for alternate external chat formats.

### 3. Dataset Validation Button

Status: initial implementation added.

**Goal:** Give users a visible "is this usable?" check before they launch training.

Build:

- Done: add API endpoints:
  - validate sandbox SFT file
  - validate corpus file or corpus folder
- Done: report:
  - file count
  - record/document count
  - empty records
  - malformed JSON/JSONL lines
  - missing train/val split
  - estimated characters/tokens
- Done: add dashboard buttons:
  - Validate SFT Dataset
  - Validate Corpus
- Done: unsupported extension reporting for ignored corpus files
- Done: exact duplicate corpus document detection
- Done: schema hints for SFT JSONL and corpus validation reports
- Done: active-tokenizer counts when a trained tokenizer is available
- Still useful later:
  - near-duplicate detection

Suggested tests:

- Done: valid SFT JSONL
- Done: malformed SFT JSONL
- Done: bad SFT role order
- Done: empty corpus
- Done: corpus document counts and split-folder warnings
- Done: corpus train/val split preview and write actions for editor content
- Done: corpus trainer-document inspection matching `nanochat.dataset` extraction
- Done: corpus inspection statistics for file types, extracted document counts, token estimates, duplicates, and very-long documents
- Still useful later: parquet validation behavior without `pyarrow`

## Phase 2: Preflight Checks Before Jobs

### 4. Stage Preflight System

Status: initial implementation added.

**Goal:** Prevent avoidable failed launches.

Build preflight checks for:

- Done: tokenizer train:
  - corpus exists
  - train docs > 0
- Done: base train:
  - tokenizer ready
  - corpus train docs > 0
  - save cadence warnings
- Done: base eval:
  - base checkpoint exists
  - eval target resolvable
- Done: chat SFT:
  - base checkpoint exists
  - train file exists
  - validation file optional but warned if absent
  - identity file status visible
- Done: chat RL:
  - SFT checkpoint exists
- Done: chat eval:
  - SFT/RL checkpoint exists
- Still useful later:
  - hardware-fit memory warnings
  - supported/ignored file extension detail
  - command diff before resume

Implementation shape:

- Done: add a pure backend preflight helper that returns structured checks.
- Done: add `POST /api/dashboard/jobs/preflight`.
- Done: in the GUI, run checks before launch with clear warnings.
- Done: allow launch with warnings, block hard failures.

Suggested files:

- New `nanochat/preflight.py`
- `scripts/chat_web.py`
- `nanochat/dashboard.html`

## Phase 3: Better Command And Job Transparency

### 5. Command Preview Before Launch

Status: initial implementation added.

**Goal:** Let users see/copy the exact CLI command before starting a job.

Build:

- Done: add an API endpoint that resolves job params to command without launching.
- Done: add "Preview Command" beside the main job launch buttons.
- Done: include local/offline environment notes used by dashboard jobs.
- Done: include preflight results in the preview response.
- Done: add explicit copy-to-clipboard for selected, previewed, and resume-preview commands.
- Done: add command diff for resumed jobs.

Suggested endpoint:

- Done: `POST /api/dashboard/jobs/preview`

### 6. Resume Preview And Safer Pause

Status: initial implementation added.

**Goal:** Make pause/resume less mysterious.

Build:

- Done: before pause, show/confirm:
  - latest checkpoint step
  - whether pause can resume
  - `save_every`
- Done: before resume, show:
  - source job
  - checkpoint tag
  - resume step
  - final command
  - command flag diff
- Done: selected job logs can be searched/filtered, and the visible tail can be copied.
- Done: failed job snapshots include a shared backend diagnosis with likely issue, detail, next step, and short evidence.
- Done: dashboard job records/logs persist to `builder_logs/jobs.json` and reload after server restart.
- Done: persistent dashboard job history is bounded to recent jobs, with active jobs retained first.
- Done: `pyarrow` parquet support is declared as the `parquet` extra, with docs and runtime errors pointing to `uv sync --extra parquet`.
- Still useful later:
  - persistence of preview decisions in activity log
  - job-history search/export
  - full-log opening, evidence highlighting, and safe direct repair actions

## Phase 4: CLI/GUI Alignment

### 7. Builder CLI Wrapper

Status: initial implementation added.

**Goal:** Expose useful dashboard-only actions from CLI.

Possible command:

```bash
python -m nanochat.builder status
python -m nanochat.builder corpus validate
python -m nanochat.builder corpus convert --input notes.md --output train/notes.parquet --mode markdown-sections
python -m nanochat.builder sft validate assistant_sandbox/chat_train.jsonl
python -m nanochat.builder job preflight --job-type chat_sft --params-json '{"train_files": "chat_train.jsonl"}'
python -m nanochat.builder job preview --job-type tokenizer_train --params-json '{"vocab_size": 16384}'
python -m nanochat.builder runtime list-models
python -m nanochat.builder runtime recommend
```

Start small:

- Done: `status`
- Done: `corpus validate`
- Done: `corpus convert`
- Done: `sft validate`
- Done: `job preflight`
- Done: `job preview`
- Done: `runtime list-models`
- Done: `runtime recommend`
- Still useful later:
  - job launch from CLI
  - runtime start/stop/status against a running managed process

### 8. Report Generation In GUI

Status: initial implementation added.

**Goal:** Surface existing `nanochat.report` commands.

Build:

- Done: Report panel
- Done: Generate report
- Done: Reset report
- Done: Show output paths and markdown preview
- Still useful later: direct open/copy actions for generated artifacts

## Phase 5: Runtime And Chat Quality

### 9. Chat Transcript Export

**Goal:** Turn Conversation Lab work into reusable training data.

Status: initial implementation.

Done:

- Save full Conversation Lab transcript to sandbox JSON
- Export full Conversation Lab transcript as an SFT JSONL row
- Export selected turns rather than the full transcript
- Preview the SFT row before appending
- Validate role/content structure before SFT export
- Load saved transcript JSON back into Conversation Lab

Still useful later:

- Add transcript import for alternate external chat formats.

### 10. Assistant Action Approval Mode

**Goal:** Reduce risk when the local GGUF assistant writes files or launches jobs.

Status: initial implementation.

Done:

- Add review mode toggle in the cockpit and Conversation Lab.
- Pause before assistant file writes, deletes, corpus copies, settings changes, job launches, and job stops.
- Show proposed path/content preview or exact job command/preflight before approval.
- Execute the exact approved action payload, then continue the assistant loop from the tool result.

Still useful later:

- Replace browser confirm prompts with a richer in-page approval panel.
- Add file diffs against existing content instead of a simple content preview.
- Store approved/denied action history with dedicated filters.

## Phase 6: Dashboard UX And Beginner Guidance

### 11. First-Run Readiness Checklist

**Goal:** Give users one obvious checklist for what is ready, what blocks training, and what to do next.

Status: initial implementation.

Done:

- Add structured readiness payload to dashboard bootstrap.
- Add `GET /api/dashboard/readiness`.
- Check local corpus train files, identity data, tokenizer, base/SFT/RL checkpoints, SFT train file, helper GGUF model availability, and local runtime readiness.
- Show the checklist in the Builder Snapshot panel with blocker/warning/ready status chips and a computed next step.

Still useful later:

- Add direct "fix this" links that scroll to the relevant panel.
- Add readiness history so users can see what changed after refresh.
- Include richer hardware/memory fit checks once stage memory estimation is available.

### 12. Named Run Profiles

**Goal:** Let users choose a run size deliberately instead of only relying on hardware-fit defaults.

Status: initial implementation.

Done:

- Add built-in run profiles: Tiny Smoke Test, Laptop Overnight, GPU Prototype, and Serious Run.
- Expose run profiles in dashboard bootstrap and `GET /api/dashboard/run-profiles`.
- Add dashboard controls to apply a selected profile without changing the assistant design.
- Add browser-local custom profile saving from the current form values.

Still useful later:

- Add delete/rename/import/export controls for custom profiles.
- Warn when a selected profile appears too large for detected hardware.
- Add profile diff/preview before applying over current form values.

### 13. Advanced Control Validation

**Goal:** Catch risky or invalid training settings before a launch spends time failing.

Status: initial implementation.

Done:

- Add backend validation for high-risk job parameters.
- Validate batch sizes, total batch divisibility, sequence length, eval tokens, save cadence, FP8/device compatibility, learning rates, and ratio/fraction fields.
- Expose `POST /api/dashboard/jobs/validate`.
- Include validation output in command preview and block job launch on validation errors.
- Surface validation warnings alongside preflight warnings in the dashboard launch flow.

Still useful later:

- Add persistent inline field-level hints beside each input.
- Add hardware-specific memory estimates instead of conservative generic warnings.
- Add one-click fixes for common issues such as total batch smaller than device batch.

### 14. Path Resolution UX

**Goal:** Make path fields explicit about whether they resolve from the repo, `local_corpus`, `assistant_sandbox`, the cache, or an absolute location.

Status: initial implementation.

Done:

- Add `POST /api/dashboard/paths/resolve`.
- Resolve sandbox-relative, corpus-relative, SFT dataset, corpus directory, identity, runtime model, and absolute path inputs.
- Include path hints in job command preview responses.
- Add a dashboard Path Resolution panel showing the current important path fields and their resolved scopes.

Still useful later:

- Add direct field-level badges beside each path input.
- Add one-click conversion between sandbox-relative and absolute paths for SFT forms.
- Add stricter validation for path fields that should never accept absolute paths.

## Phase 7: Testing And Reliability

### 15. Command Parity Tests

**Goal:** Catch drift between dashboard job command generation and the underlying script CLI arguments.

Status: initial implementation.

Done:

- Add a dashboard job parameter manifest keyed by job type.
- Add generated tests that parse script `argparse.add_argument()` calls and compare long flags against dashboard-supported params.
- Allow explicit local-only omissions such as `base_eval --hf-path`.
- Add a smoke assertion that supported params emit expected command flags.
- Extend parity coverage to the dashboard HTML form field names.

Still useful later:

- Fail with a richer report grouping missing, extra, and intentionally unsupported fields.
- Add parity checks to the builder CLI wrapper if job launch support is added there.

### 16. Dashboard Browser Smoke Tests

Status: initial implementation.

Done:

- Add optional `browser-tests` extra with Playwright.
- Add mocked-browser smoke tests for dashboard bootstrap rendering.
- Add selected-job diagnosis/log-filter smoke coverage.
- Add base-train command preview form serialization smoke coverage.
- Add sandbox and corpus file editor save smoke coverage.
- Add runtime start/stop control smoke coverage.
- Add Conversation Lab local-runtime chat send, transcript JSON save, and SFT JSONL export smoke coverage.
- Add launch-flow smoke coverage for form validation blockers, preflight blockers, and successful mocked job creation/selection.

Run:

```bash
uv sync --extra browser-tests
uv run playwright install chromium
uv run python -m pytest tests/test_dashboard_playwright.py
```

Still useful later:

- Add future assistant approval panel coverage.

## Completed First Work Slice

Completed:

1. Added backend conversion helpers for corpus parquet creation.
2. Added tests for conversion modes and parquet writing.
3. Added API endpoints for parquet preview/write.
4. Added dashboard UI controls for "Create Parquet Corpus".
5. Added corpus validation endpoints and dashboard buttons.

Why this was first:

- It solves a real current gap.
- It improves beginner success before training.
- It is mostly local/backend logic with focused tests.
- It lays groundwork for SFT export and preflight checks.

## Success Criteria For The First Slice

- Done: a user can paste plain text or markdown and create `local_corpus/train/*.parquet`.
- Done: the GUI previews rows before writing.
- Done: the backend rejects malformed structured input with useful errors.
- Done: existing corpus text/json/jsonl behavior still works.
- Done: tests cover the new conversion helper and API.
- Done: docs identify corpus parquet and SFT parquet as separate concepts.
