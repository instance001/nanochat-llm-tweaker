# CLI / GUI Function Map

This file maps the current command-line surfaces to the dashboard surfaces so gaps can be discussed before implementation work begins.

## CLI Functions

These are direct command-line entry points currently exposed by scripts or modules.

| CLI entry point | Main function | Key parameters |
| --- | --- | --- |
| `python -m scripts.tok_train` | Train a tokenizer from the local corpus. | `--corpus-dir`, `--max-chars`, `--doc-cap`, `--vocab-size` |
| `python -m scripts.tok_eval` | Evaluate tokenizer compression/sample behavior against the local corpus. | `--corpus-dir`, `--max-corpus-docs` |
| `python -m scripts.base_train` | Train a base model from the local corpus. | `--corpus-dir`, `--depth`, `--aspect-ratio`, `--head-dim`, `--max-seq-len`, `--window-pattern`, `--device-batch-size`, `--total-batch-size`, `--num-iterations`, `--target-flops`, `--target-param-data-ratio`, `--embedding-lr`, `--unembedding-lr`, `--matrix-lr`, `--scalar-lr`, `--weight-decay`, `--adam-beta1`, `--adam-beta2`, `--warmup-ratio`, `--warmdown-ratio`, `--final-lr-frac`, `--resume-from-step`, `--eval-every`, `--eval-tokens`, `--core-metric-every`, `--core-metric-max-per-task`, `--sample-every`, `--save-every`, `--model-tag`, `--run`, `--device-type`, `--fp8`, `--fp8-recipe` |
| `python -m scripts.base_eval` | Evaluate a base model. | `--eval`, `--hf-path`, `--model-tag`, `--step`, `--max-per-task`, `--device-batch-size`, `--split-tokens`, `--device-type`, `--corpus-dir` |
| `python -m scripts.chat_sft` | Supervised fine-tune a chat model. | `--model-tag`, `--model-step`, `--load-optimizer`, `--resume-from-step`, `--train-files`, `--val-files`, `--identity-file`, `--include-identity`, `--identity-repeats`, `--max-seq-len`, `--device-batch-size`, `--total-batch-size`, `--num-iterations`, `--embedding-lr`, `--unembedding-lr`, `--matrix-lr`, `--init-lr-frac`, `--warmup-ratio`, `--warmdown-ratio`, `--final-lr-frac`, `--eval-every`, `--eval-tokens`, `--chatcore-every`, `--save-every`, `--run`, `--device-type` |
| `python -m scripts.chat_rl` | Reinforcement learning stage for chat models. | `--model-tag`, `--model-step`, `--resume-from-step`, `--num-epochs`, `--device-batch-size`, `--examples-per-step`, `--num-samples`, `--max-new-tokens`, `--temperature`, `--top-k`, `--embedding-lr`, `--unembedding-lr`, `--matrix-lr`, `--weight-decay`, `--init-lr-frac`, `--eval-every`, `--eval-examples`, `--save-every`, `--run`, `--device-type` |
| `python -m scripts.chat_eval` | Evaluate SFT or RL chat checkpoints on built-in tasks. | `--source`, `--task-name`, `--temperature`, `--max-new-tokens`, `--num-samples`, `--top-k`, `--batch-size`, `--model-tag`, `--step`, `--max-problems`, `--device-type` |
| `python -m scripts.chat_cli` | One-shot or interactive chat with a trained nanochat checkpoint. | `--source`, `--model-tag`, `--step`, `--prompt`, `--temperature`, `--top-k`, `--device-type` |
| `python -m scripts.chat_web` | Start the combined chat and builder web server. | `--port`, `--host`, `--num-gpus`, `--source`, `--model-tag`, `--step`, `--temperature`, `--top-k`, `--max-tokens`, `--runtime-autostart`, `--runtime-model`, `--runtime-port`, `--runtime-device-strategy`, `--device-type` |
| `python -m nanochat.dataset` | Inspect local corpus contents. | `--data-dir`, `--split`, `--show-docs` |
| `python -m nanochat.builder status` | Print local builder status. | `--corpus-dir`, `--sandbox-dir` |
| `python -m nanochat.builder corpus validate` | Validate a corpus file or directory. | optional path, default local corpus |
| `python -m nanochat.builder corpus convert` | Convert text/markdown/JSON/JSONL input into corpus parquet. | `--input`, `--output`, `--corpus-dir`, `--mode`, `--write-mode`, `--source` |
| `python -m nanochat.builder sft validate` | Validate an SFT JSONL conversation file. | path |
| `python -m nanochat.builder job preflight` | Run launch preflight for a dashboard job from CLI. | `--job-type`, `--params-json`, `--params-file`, `--base-dir`, `--corpus-dir`, `--sandbox-dir` |
| `python -m nanochat.builder job preview` | Preview a dashboard job command and preflight from CLI. | `--job-type`, `--params-json`, `--params-file`, `--base-dir`, `--corpus-dir`, `--sandbox-dir` |
| `python -m nanochat.builder runtime list-models` | List discovered local GGUF/GGML models. | `--repo-root` |
| `python -m nanochat.builder runtime recommend` | Print runtime bundle status and recommended start config. | `--repo-root` |
| `python -m nanochat.report` | Generate or reset the nanochat report. | `generate`, `reset` |

## GUI Functions

These are dashboard-facing functions exposed through `nanochat/dashboard.html` and `scripts/chat_web.py`.

### Builder State And Monitoring

| GUI function | Backend route or mechanism | Notes |
| --- | --- | --- |
| Refresh State | `GET /api/dashboard/bootstrap` | Loads builder state, jobs, chat status, runtime status, ECG, sandbox, corpus, activity, and benchmark history. |
| ECG Window | `GET /api/dashboard/ecg` | Live CPU/GPU/app activity signal. |
| Process Log | `GET /api/activity/status` | Shows recent job, runtime, chat, corpus, sandbox, and assistant activity. |
| Benchmark History | `GET /api/dashboard/benchmarks` | Summarizes recorded benchmark/job metrics. |
| Fit To Hardware | client-side form fill from `builder.hardware_recommendations` | Applies detected hardware recommendations to dashboard forms. |

### Design And Identity

| GUI function | Backend route or mechanism | Notes |
| --- | --- | --- |
| Use Truth-First Preset | client-side preset selection from bootstrap state | Fills design and recipe defaults. |
| Save Design | `POST /api/dashboard/designs` | Saves a reusable assistant design and generated starter identity examples. |
| Publish as Active Identity File | `POST /api/dashboard/designs/{slug}/publish` | Writes the active identity JSONL file used by SFT. |
| Delete Design | `DELETE /api/dashboard/designs/{slug}` | Removes a saved design. |
| Draft With GGUF | `POST /api/dashboard/designs/draft` | Uses the managed local GGUF runtime to draft a design JSON payload. |

### Training And Evaluation Jobs

| GUI function | Backend job type | CLI command generated |
| --- | --- | --- |
| Train Tokenizer | `tokenizer_train` | `python -m scripts.tok_train` |
| Run Tokenizer Eval | `tokenizer_eval` | `python -m scripts.tok_eval` |
| Launch Base Training | `base_train` | `python -m scripts.base_train` |
| Run Base Eval | `base_eval` | `python -m scripts.base_eval` |
| Run Benchmark | `benchmark_eval` | `python -m scripts.base_eval` with benchmark-oriented defaults |
| Launch Chat SFT | `chat_sft` | `python -m scripts.chat_sft` |
| Launch Chat RL | `chat_rl` | `python -m scripts.chat_rl` |
| Run Chat Eval | `chat_eval` | `python -m scripts.chat_eval` |
| Auto Tune Next Run | `POST /api/dashboard/autotune` | Uses benchmark history to recommend form settings. |
| Pause Job | `POST /api/dashboard/jobs/{job_id}/pause` | Terminates resumable jobs after a checkpoint is available. |
| Resume Job | `POST /api/dashboard/jobs/{job_id}/resume` | Starts a new job with `--resume-from-step`. |
| Stop Job | `POST /api/dashboard/jobs/{job_id}/stop` | Terminates the selected background process. |
| Copy Resume To Form | client-side checkpoint metadata copy | Copies latest checkpoint step into relevant form fields. |
| Use Checkpoint In Eval/Load | client-side checkpoint metadata copy | Copies checkpoint tag/step into eval or chat load forms. |

### Chat And Runtime

| GUI function | Backend route or mechanism | Notes |
| --- | --- | --- |
| Load Chat Model | `POST /api/dashboard/chat/load` | Loads internal nanochat SFT/RL checkpoint workers. |
| Load Latest SFT | `POST /api/dashboard/chat/load` | Shortcut for latest SFT checkpoint. |
| Load Latest RL | `POST /api/dashboard/chat/load` | Shortcut for latest RL checkpoint. |
| Start Local Runtime | `POST /api/runtime/start` | Starts bundled `llama-server.exe` with selected GGUF settings. |
| Stop Local Runtime | `POST /api/runtime/stop` | Stops managed llama.cpp runtime. |
| Run Quick Prompt | `/chat/completions` or `POST /api/runtime/chat` | Dashboard equivalent of one-shot `chat_cli.py -p`, with provider selection. |
| Conversation Lab Send | `/chat/completions`, `POST /api/runtime/chat`, or `POST /api/runtime/assist` | Multi-turn test bench using internal nanochat or local GGUF runtime. |
| Apply Cockpit To Bench | client-side setting copy | Copies cockpit provider/sampling/system prompt controls into quick prompt and chat controls. |
| Activate Backend Profile | client-side helper plus chat/runtime routes | Switches selected assistant backend profile. |
| Reset Saved Controls | client-side local storage reset | Clears saved cockpit/runtime settings. |

### Sandbox, Corpus, And Assistant Actions

| GUI function | Backend route or mechanism | Notes |
| --- | --- | --- |
| List Sandbox Files | `GET /api/sandbox/files` | Lists files under `assistant_sandbox`. |
| Load Sandbox File | `GET /api/sandbox/file` | Reads a scoped sandbox file. |
| Save Sandbox File | `POST /api/sandbox/write` | Writes a scoped sandbox file. |
| Delete Sandbox File | `POST /api/sandbox/delete` | Deletes a scoped sandbox file. |
| Save Last Assistant Reply | client-side plus `POST /api/sandbox/write` | Saves last reply to a sandbox file. |
| Preview/Export SFT Dataset | `POST /api/sandbox/sft/export` | Previews normalized JSONL, parquet rows, or deterministic train/val splits; writes normalized JSONL, parquet, or split JSONL files. |
| List Corpus Files | `GET /api/corpus/files` | Lists files under `local_corpus`. |
| Load Corpus File | `GET /api/corpus/file` | Reads text/JSON/JSONL or previews parquet files. |
| Save Corpus File | `POST /api/corpus/write` | Writes text/JSON/JSONL or structured parquet records. |
| Delete Corpus File | `POST /api/corpus/delete` | Deletes a scoped corpus file. |
| Copy Sandbox To Corpus | `POST /api/corpus/copy-from-sandbox` | Copies sandbox content into corpus, with parquet conversion support. |
| Assistant Actions | `POST /api/runtime/assist` | Local GGUF can call scoped actions for builder state, activity, benchmark history, autotune, corpus files, sandbox files, SFT drafting, job launch, job status, and job stop. |

## GUI Does Not Match CLI Functions

### CLI Function Exists But GUI Is Missing Or Partial

| CLI function | GUI status | Gap |
| --- | --- | --- |
| `scripts.base_eval --hf-path` | Missing from GUI | The GUI only targets local nanochat checkpoints. It does not expose Hugging Face model evaluation. |
| `scripts.chat_cli` interactive mode | Partial | The GUI has Quick Prompt and Conversation Lab, but no terminal-style interactive CLI session. |
| `scripts.chat_cli --prompt` | Partial | Quick Prompt is equivalent in spirit, but supports both internal nanochat and local GGUF runtime, while `chat_cli.py` only loads nanochat checkpoints. |
| `scripts.chat_web --host` | Missing from launcher GUI | The dashboard can be launched with host through CLI, but the ChattyCog launcher fixes the hosted URL path. |
| `scripts.chat_web --runtime-autostart` | Partial | CLI exposes autostart on/off. GUI exposes start/stop runtime after server boot, but not the server-start autostart flag. |
| `scripts.chat_web --runtime-model`, `--runtime-port`, `--runtime-device-strategy` | Partial | GUI can manage equivalent runtime settings after boot; launcher also passes runtime port/device strategy. |
| Shell run scripts in `runs/` | Missing from GUI | Speedrun, miniseries, scaling laws, and CPU sample shell scripts are not dashboard job types. |

### GUI Function Exists But CLI Is Missing Or Indirect

| GUI function | CLI status | Gap |
| --- | --- | --- |
| Save / publish / delete assistant designs | Indirect Python functions only | There is no first-class CLI for managing builder designs or publishing identity files. |
| Draft design with local GGUF | GUI/API only | No direct CLI wrapper around `POST /api/dashboard/designs/draft`. |
| Hardware-fit form recommendations | Python helper only | No CLI command prints/applies `recommend_forms_for_hardware`. |
| Dashboard job queue | GUI/API only | CLI scripts run directly; they do not get dashboard job IDs, persisted log tails, pause/resume buttons, or benchmark-history capture unless launched through the dashboard server. |
| Pause / resume selected job | GUI/API only | Resume flags exist in training scripts, but pause/resume orchestration is a dashboard job-manager feature. |
| Benchmark history and auto-tune | GUI/API only | No standalone CLI command exposes benchmark history recommendations. |
| Managed llama.cpp runtime | GUI/API only | The bundled runtime manager is exposed through dashboard APIs, not as a standalone CLI. |
| Cockpit controls | GUI-only | Backend profile, action mode, and saved operator controls live in browser state and dashboard requests. |
| Conversation Lab with Assistant Actions | GUI/API only | CLI chat does not expose the local assistant action loop. |
| Scoped sandbox editor | GUI/API only | No standalone CLI for sandbox list/read/write/delete. |
| Scoped corpus editor with parquet write support | GUI/API only | Corpus reading exists for training, but GUI/API adds scoped editing and parquet creation. |
| Activity log view | GUI/API only | No standalone CLI command tails or summarizes `builder_logs/activity.jsonl`. |
| ECG window | GUI/API only | No standalone CLI command shows the activity monitor snapshot. |

### Naming Or Behavior Differences To Watch

| Area | Difference |
| --- | --- |
| Base benchmark | GUI `benchmark_eval` maps to `scripts.base_eval`, not a separate CLI script. |
| Local-only defaults | Dashboard-launched jobs set local/offline environment variables consistently. Direct CLI use depends on the user environment or script behavior. |
| Checkpoint resume | CLI exposes `--resume-from-step`; GUI infers the latest checkpoint and launches a new resumed job. |
| Chat providers | CLI chat uses internal nanochat checkpoints. GUI chat can use either internal nanochat or the managed local GGUF runtime. |
| SFT file paths | CLI accepts file paths directly. Assistant actions coerce relative SFT paths into `assistant_sandbox`; dashboard forms may contain absolute or copied paths. |
| Corpus parquet editing | CLI training can read parquet when `pyarrow` is installed. GUI/API can also create and preview parquet records. |
| Defaults | GUI defaults are conservative local-builder presets. Upstream CLI script defaults may be larger or more research-oriented. |

## Gaps, Missing Parts, And Extension Ideas

This section is a working backlog of places where functions exist but are incomplete, awkward, or could be extended.

### LLM Types Currently Supported Vs Not Supported

| Type | Current support | Notes |
| --- | --- | --- |
| Text-only base LLM training | Supported | `scripts.base_train` trains a GPT-style text model from local text extracted from text/code/JSON/JSONL/parquet corpus files. |
| Text-only chat SFT | Supported | `scripts.chat_sft` fine-tunes text chat behavior from local JSONL conversation files plus optional identity examples. |
| Text-only chat RL | Supported | `scripts.chat_rl` runs the current RL stage for chat checkpoints. |
| Text-only chat inference | Supported | Internal nanochat checkpoints can be loaded through `chat_cli.py`, `chat_web.py`, Quick Prompt, and Conversation Lab. |
| Local GGUF text helper/runtime | Supported | The dashboard can start bundled llama.cpp `llama-server.exe` for local `.gguf` / `.ggml` helper models. This is runtime/inference support, not nanochat training support. |
| Code/text corpus training | Supported as text | Source code files are treated as text documents for tokenizer/base training. There is no separate code-model architecture. |
| Structured text datasets | Partially supported | JSON, JSONL, and parquet can be read as sources of text. SFT expects conversation JSONL. |
| Multimodal vision-language models | Not supported for training | The bundled runtime includes some multimodal-looking llama.cpp binaries, but the dashboard/training pipeline does not currently handle images, vision encoders, image tokens, or VLM datasets. |
| Image generation models | Not supported | No diffusion/image model training or inference pipeline is present. |
| Audio/speech models | Not supported | No ASR, TTS training, audio tokenization, or audio dataset flow is present. |
| Embedding models | Not supported | No embedding-specific training, evaluation, vector indexing, or retrieval benchmark path is present. |
| Reranker/classifier models | Not supported | No classification/reranking fine-tune path is exposed. |
| Tool-use/function-calling models | Partially supported as behavior data | The dashboard's local GGUF assistant can use dashboard actions through a custom text protocol, but nanochat training does not currently have a first-class tool-call schema/eval loop. |
| Retrieval-augmented generation | Partially supported as prompt context | Conversation Lab can include selected sandbox/corpus files as context, but there is no vector store, retriever, chunker, or RAG evaluation pipeline. |
| Long-context specialization | Partially supported | Sequence length and runtime context size are configurable, but there is no dedicated long-context curriculum, eval suite, or memory optimization workflow. |
| Mixture-of-experts models | Not supported | Current nanochat model path is a dense GPT-style transformer. |
| Quantization of trained nanochat checkpoints | Not supported in dashboard | The runtime can consume external GGUFs, but there is no GUI flow to convert nanochat checkpoints to GGUF or quantize them. |
| Distributed/cloud training | Not the intended local-builder path | Upstream nanochat has distributed research workflows, but this fork's dashboard is oriented around local/offline runs. |

### Dataset And Corpus Creation

| Area | Current state | Improvement idea |
| --- | --- | --- |
| SFT dataset creation | The GUI and assistant actions draft SFT data as JSONL conversation files under `assistant_sandbox`, and the Sandbox Editor can preview, normalize, export, and split those files. Conversation Lab can export full or selected turns as SFT JSONL. | Add transcript import for alternate external chat formats and more ergonomic preview modals if the simple metadata preview becomes too cramped. |
| SFT parquet output | Initial GUI/API export exists for validated SFT conversations, writing one conversation per parquet row with serialized messages and summary columns. The Sandbox Editor can load exported parquet files as preview-only sample rows. `pyarrow` is available through the `parquet` extra. | Add optional schema variants for downstream tools. |
| Corpus parquet creation | Initial GUI/API converter now previews and writes `.parquet` files from paragraphs, lines, markdown sections, JSONL objects, or JSON object/array input. | Extend with richer validation, append-mode UI, saved conversion presets, and larger dataset statistics. |
| Dataset validation | Initial GUI/API validation exists for sandbox SFT JSONL files and corpus paths/files, including row/document counts, malformed line errors, split-folder warnings, unsupported/ignored file reporting, exact duplicate detection, schema hints, rough token estimates, and active-tokenizer counts when a tokenizer is available. | Extend with near-duplicate detection if needed. |
| Train/val split creation | GUI/API can split SFT JSONL files and current Corpus Editor content into deterministic train/val outputs with seed and ratio controls. | Add whole-folder corpus rebalancing, split manifests, and richer preview modals. |
| Dataset preview depth | GUI can inspect extracted documents using the same `nanochat.dataset` reader as tokenizer/base training. | Add side-by-side raw file vs extracted document comparison if needed. |
| Dataset statistics | Corpus inspection now reports file type breakdown, document/character/token estimates, duplicate counts, average/max document length, longest documents, and very-long-document warnings. | Add near-duplicate detection and richer per-file trend views if needed. |

### CLI And GUI Alignment

| Area | Current state | Improvement idea |
| --- | --- | --- |
| CLI for dashboard-only features | Initial `python -m nanochat.builder` exists for status, corpus validation/conversion, SFT validation, job preflight, job command preview, runtime model listing, and runtime recommendation. Designs, live runtime start/stop/status, activity logs, ECG, sandbox editing, job launch, and auto-tune are still GUI/API-only. | Extend the builder CLI incrementally around the shared backend helpers. |
| GUI for report generation | Initial dashboard report panel exists for status, generate, reset, output paths, and markdown preview. | Add direct open/copy actions for generated artifacts if needed. |
| GUI for corpus inspection CLI | `python -m nanochat.dataset` can inspect documents by split, and the dashboard now has a matching trainer-document inspection action with split, show-docs, and max-character controls. | Extend with richer per-file extraction stats if needed. |
| GUI for run scripts | `runs/speedrun.sh`, `runs/miniseries.sh`, `runs/scaling_laws.sh`, and `runs/runcpu.sh` are not represented in the dashboard. | Decide whether these should stay research-only or become optional advanced dashboard recipes. |
| GUI for `base_eval --hf-path` | Direct CLI can evaluate a Hugging Face model path. | Because this fork is local-first/offline by default, either hide this intentionally in docs or add an advanced local-path-only equivalent. |
| Export generated commands | GUI shows selected, previewed, and resume-preview commands, can copy them to the clipboard, and shows command diffs for resumed jobs. | Add export/download for command history if needed. |

### Job Management And Resume

| Area | Current state | Improvement idea |
| --- | --- | --- |
| Pause behavior | Pause terminates a running process and marks it paused only when a checkpoint exists. | Show a stronger warning before pause when no checkpoint has been saved yet, including current `save_every` and latest checkpoint step. |
| Stage preflight | Initial backend/API/dashboard preflight exists for launch blockers like missing corpus, tokenizer, checkpoints, SFT files, and SFT file format mistakes. | Extend with hardware-fit memory estimates, ignored-file reporting, and richer stage-specific repair links. |
| Resume setup | GUI can infer latest checkpoint, preview checkpoint tag/step/source job, show changed resume parameters and command diff, copy the command, and launch a resumed job. | Persist preview decisions in activity history if needed. |
| Job persistence | Dashboard job records and log tails are persisted to `builder_logs/jobs.json`, bounded to recent history, and reloaded on server startup. Previously running/queued jobs are marked stopped because process handles cannot be reattached after restart. | Add richer archived-job search/export if job history grows large. |
| Job log search | Selected job logs can be searched/filtered in the dashboard, and the visible tail can be copied. | Add open-full-log and inline highlighting for diagnosis evidence. |
| Failed-job diagnosis | Job snapshots now include structured likely-failure summaries for missing dependencies, OOM, empty corpus, missing tokenizer, identity file, checkpoints, and missing paths. The dashboard renders the likely issue, why it failed, next step, and evidence. | Extend signatures as new failure modes show up, and add direct repair actions where safe. |

### Runtime And Chat

| Area | Current state | Improvement idea |
| --- | --- | --- |
| Managed llama.cpp CLI | CLI now supports listing GGUFs and printing recommended runtime config. GUI/API still owns live start/stop/status. | Add standalone start/stop/status and smoke testing if a CLI-managed runtime becomes useful. |
| Runtime model import | The runtime scans known folders for GGUF/GGML files. | Add a GUI action to copy or register a model path into `assistant_models` with metadata. |
| Cockpit profiles | Cockpit settings are browser-local and manually applied. | Persist named cockpit profiles with import/export and per-model defaults. |
| Assistant action safety | GUI/API can pause mutating Assistant Actions for approval before writes, deletes, corpus copies, settings changes, job launches, and job stops. | Replace confirm prompts with an in-page approval panel and add richer file diffs before write/append. |
| Chat transcript export | GUI/API can save full or selected Conversation Lab turns to sandbox JSON, load saved transcript JSON back into Conversation Lab, preview SFT JSONL rows, or append selected/full user-assistant turns as SFT JSONL. | Add transcript import for alternate external chat formats. |
| Quick Prompt parity | Quick Prompt is similar to `chat_cli.py -p` but not exactly the same. | Add an option to show the exact equivalent CLI command when using internal nanochat. |

### Dashboard UX And Beginner Guidance

| Area | Current state | Improvement idea |
| --- | --- | --- |
| First-run readiness | GUI/API now expose a readiness checklist for helper model, corpus, tokenizer, base/SFT/RL checkpoints, SFT files, and runtime. | Add direct fix links, readiness history, and richer hardware/memory checks. |
| Stage dependencies | Preflight checks block later-stage launches when required artifacts are missing, with clear stage-specific messages. | Add direct "fix this first" links that scroll to the relevant dashboard panel. |
| Form defaults | Hardware-fit recommendations and named run profiles can fill forms. Custom profiles can be saved browser-locally from current form values. | Add delete/rename/import/export controls, profile diffs, and hardware-fit warnings for oversized profiles. |
| Advanced controls | GUI/API validate high-risk job params before launch and include validation output in command preview. | Add persistent field-level hints, hardware-specific memory estimates, and one-click fixes for common invalid settings. |
| Path handling | GUI/API expose path resolution hints for sandbox, corpus, SFT, identity, runtime model, and absolute paths, and command preview includes path hints. | Add field-level badges, path conversion buttons, and stricter validation for fields that should not accept absolute paths. |

### Testing And Reliability

| Area | Current state | Improvement idea |
| --- | --- | --- |
| Dashboard HTML coverage | Initial opt-in Playwright smoke tests cover dashboard bootstrap rendering, selected-job failure/log filtering, base-train command preview form serialization, launch validation/preflight/success flows, sandbox/corpus file editor saves, runtime start/stop controls, local-runtime chat send, transcript JSON save, and SFT JSONL export with mocked API responses. | Extend Playwright coverage to the future assistant approval panel and more runtime edge cases. |
| Command parity tests | Generated tests compare script argparse long flags and dashboard HTML form field names against the dashboard job parameter manifest. | Produce a richer missing/extra/unsupported report if parity failures become hard to read. |
| Dataset conversion tests | Corpus/SFT helpers have targeted tests, including corpus parquet conversion and SFT parquet export. | Add tests for split creation, export preview UI behavior, and richer validation report behavior. |
| Runtime tests | Runtime parsing and config recommendation tests exist. | Add tests for model discovery edge cases, preferred device selection, port reuse, and CLI wrapper if added. |
| Windows launcher tests | Launcher scripts are manually inspected. | Add a lightweight script check or documented smoke procedure for `.cmd` / `.ps1` launcher behavior. |

### Packaging And Distribution

| Area | Current state | Improvement idea |
| --- | --- | --- |
| Generated files | Runtime logs, activity logs, persistent dashboard job state, local corpus files, assistant sandbox JSONL, model folders, caches, and temporary launcher logs are ignored in both root and nested `.gitignore` files. | Add a generated-files reference section if distribution docs need one. |
| Runtime binaries | Windows llama.cpp runtime is bundled under `runtime/windows`. | Document update process, version source, expected files, and license notes for runtime refreshes. |
| Optional dependencies | Parquet support depends on `pyarrow`, declared as the `parquet` extra in `pyproject.toml`. Runtime errors and docs point users to `uv sync --extra parquet`. | Add installer UI hints if users commonly hit missing optional dependencies from the dashboard. |
| ChattyCog integration | Outer wrapper starts the nested dashboard through fixed ports. | Add collision handling notes or a dynamic-port strategy if ChattyCog supports it. |
