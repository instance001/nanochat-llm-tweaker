# Local Builder User Manual

## Read This First

This manual is written with a zero-knowledge assumption.

That means it assumes you may be new to:

- local LLM runtimes
- tokenizer training
- base-model training
- SFT datasets
- ChattyCog module hosting

If you are feeling unsure, that is expected. Start small, follow the workflow in order, and treat this as a practical workbench rather than a magic one-click trainer.

If you want the most beginner-friendly setup guide with dependency commands, search suggestions, and an end-to-end walkthrough of each dashboard area, read:

- [ZERO_EXPERIENCE_END_TO_END_GUIDE.md](ZERO_EXPERIENCE_END_TO_END_GUIDE.md)

## What This Is

This project is a local-only LLM builder and assistant workspace.

It lets you:

- run a local GGUF chat model through the bundled `llama.cpp` runtime
- prepare local training data
- train a tokenizer from local files
- train a base model from local files
- fine-tune the model with local chat JSONL files
- test progress in a browser dashboard
- use a local assistant to help draft data, review state, and launch local jobs

This fork is designed around one rule:

**No cloud path is required for the normal builder workflow.**

The dashboard does not depend on remote dataset downloads, remote logging, or hosted inference.

## Licensing And Upstream Credit

This builder is based on Andrej Karpathy's `nanochat` project.

License boundary:

- Upstream-derived code inside this `nanochat-master/` folder follows the upstream nanochat license where applicable. See [LICENSE](LICENSE).
- The outer ChattyCog module wrapper that hosts this builder is a separate layer with its own license file and module metadata.

If you share or redistribute the packaged ChattyCog module, keep both license files:

- `nanochat-master/LICENSE`
- the outer module wrapper `LICENSE`

## What This Is Not

This is not a one-click magic model factory.

It is a practical local workbench. You still need to:

- provide your own local corpus
- review the assistant's drafts before training on them
- choose reasonable model sizes for your hardware
- iterate when the model behavior is not good enough

If you keep that mental model, the system makes sense quickly.

## The Big Picture

There are two different "LLMs" in this workspace:

1. **Local runtime assistant**
   Uses a `.gguf` model from `assistant_models` through the bundled `llama.cpp` runtime.
   This is the assistant that helps you operate the builder right away.

2. **Internal nanochat model**
   This is the model you train with the pipeline in the dashboard.
   It starts as your own tokenizer + base model + chat fine-tune checkpoints.

In plain English:

- the GGUF model helps you build
- the nanochat training pipeline builds the model you actually own and train here

## If This Is Your First Time Building Or Training An LLM

This section is for the person who is thinking:

- I have heard the words, but I do not really know what these tasks mean
- I can follow instructions, but I want to understand what I am actually doing
- I do not want the process to feel like superstition

That is a good instinct.

You do not need to become an ML researcher to use this builder, but you do need a plain-language mental model of the tasks.

### What An LLM Actually Is

An LLM is a machine-learning model that tries to predict the next piece of text based on the text that came before it.

Plain English:

- it does not store truth the way a normal document does
- it learns patterns from examples
- the quality of those examples strongly affects how it behaves later

That means:

- bad data can teach bad habits
- confused instructions can produce confused behavior
- a tiny model on a tiny dataset can still be useful, but it will not become magic

### What Training Means

Training means adjusting the model's internal numbers so it gets better at predicting text from examples.

Plain English:

- the model reads examples
- it makes guesses
- the system measures how wrong those guesses were
- then it nudges the model so the next guesses are a bit better

Training is not:

- copying a file into memory
- storing exact facts on purpose
- a guarantee that the model will always say the right thing

### What A Tokenizer Is

A tokenizer is the part that breaks text into chunks called tokens before the model sees it.

Plain English:

- the model does not read text the way a human reads whole words on a page
- it reads token pieces
- those pieces might be words, word fragments, punctuation, or common patterns

Why it matters:

- a bad tokenizer makes learning harder
- a reasonable tokenizer helps the model use its small capacity more effectively

### What Base Training Is

Base training teaches the model general language patterns from the broad corpus.

This is the stage where it learns things like:

- how sentences are shaped
- how code tends to look
- how explanations and references are usually structured

Plain English:

- this is the `general reading pile` stage
- it is not yet the `be this specific helpful assistant` stage

### What Fine-Tuning Is

Fine-tuning means taking a base model and teaching it more specific behavior with more focused examples.

In this builder, the main fine-tuning stage is Chat SFT.

Plain English:

- base training teaches general language habits
- fine-tuning teaches the model how you want it to respond in practice

This is where the model learns things like:

- how to answer like a teammate
- how to admit uncertainty
- how to follow your chosen tone and rules

### What SFT Means

SFT means supervised fine-tuning.

That sounds intimidating, but the practical meaning is simple:

- you give the model example conversations
- each conversation shows what a good response should look like
- the model gets better at copying those good response patterns

Plain English:

- SFT is teaching by example
- you are showing the model what `good` looks like in conversation form

### What RL Means Here

RL means reinforcement-learning-style tuning.

In this project, it is the more advanced stage after SFT.

Plain English:

- SFT gets the model into a usable assistant shape
- RL is for later experimentation when you want to push or refine behavior further

If you are new:

- do not treat RL as the first important step
- get tokenizer, base, and SFT working first

### What Evaluation Means

Evaluation means checking how the model is doing instead of just hoping it is better.

That can include:

- loss-style numbers
- benchmark results
- sample outputs
- quick smoke tests in chat

Plain English:

- evaluation is how you avoid guessing blindly
- it is the `show me evidence` part of the workflow

### What Tuning Means

Tuning means changing settings to make the process fit your goal or your hardware better.

Common things you tune here include:

- model size
- sequence length
- batch size
- training duration
- runtime context size
- temperature and token limits during testing

Plain English:

- tuning is not magic
- it is mostly the art of making tradeoffs on purpose

### The Main Data Types In This Builder

You are working with different kinds of data for different jobs.

#### 1. Local Corpus Data

Used for:

- tokenizer training
- base model training

Plain English:

- this is the general reading material

#### 2. Identity Data

Used for:

- teaching the assistant how it should think about its role and behavior

Plain English:

- this is the assistant charter

#### 3. Chat SFT Data

Used for:

- teaching real conversation behavior through examples

Plain English:

- this is the set of example dialogues showing how the assistant should answer

### Why Small First Runs Matter

New users often want to launch a big run immediately.

That is understandable, but usually the smarter move is:

1. prove the pipeline works
2. prove the files are valid
3. prove the runtime fits the machine
4. only then spend more time and compute

Plain English:

- first make it work
- then make it bigger

### What Usually Goes Wrong For Beginners

The most common early problems are not exotic ML failures.

They are usually:

- the corpus is missing or too messy
- the training files are malformed
- the model is too large for the machine
- the user changes too many settings at once
- the user cannot tell whether the system is idle, stalled, or actually broken

That is why this fork includes things like:

- hardware-fit defaults
- pause and resume
- the ECG window
- process logs
- guided presets

### The Best Beginner Mindset

Think of this builder as a lab bench, not a vending machine.

Good mindset:

- start small
- change one important thing at a time
- keep notes
- trust evaluation more than vibes
- review the assistant's drafted data before training on it

Best short version:

- you are not pressing a button to summon intelligence
- you are building behavior step by step from data, settings, and iteration

## Folder Map

These folders matter most:

- `assistant_models`
  Put local `.gguf` files here for the helper assistant.

- `assistant_sandbox`
  The only writable workspace exposed to the local assistant through the dashboard.
  Use it for draft chat datasets, notes, prompts, and templates.

- `local_corpus`
  Put your local training corpus here.
  The builder reads this for tokenizer training and base model training.

- `runtime/windows`
  Bundled Windows `llama.cpp` runtime files live here.

- `builder_logs`
  Local logs live here.
  The main activity log is `builder_logs/activity.jsonl`.

- `runs`
  Upstream training scripts live here, but the dashboard is the intended entry point for this local fork.

## Before You Start

You do not need deep ML knowledge to begin, but you do need a few practical things in place.

### Minimum Requirements

- Windows machine
- Python available
- a working `.venv` or a Python environment with `fastapi` and `uvicorn`
- local GGUF model files in `assistant_models`
- local text/code/document data in `local_corpus`

### Optional but Useful

- Vulkan-capable GPU for the local runtime assistant
- CUDA or MPS for faster training if your system supports it
- the `parquet` extra, installed with `uv sync --extra parquet`, only if you want to use `.parquet` corpus files or SFT parquet export
- Microsoft Visual Studio Build Tools with the Desktop C++ workload if you plan to do local Windows CPU training

### Recommended Install Profiles

Use these from `nanochat-master/`.

For most Windows machines, including AMD Windows boxes:

```powershell
uv sync --extra cpu --extra parquet
```

This gives you CPU PyTorch training plus parquet dataset support.

On AMD Windows, this check is expected:

```powershell
uv run python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Expected result:

```text
2.9.1+cpu
CUDA: False
```

That does not mean the setup is broken. CUDA is NVIDIA-only. AMD Windows should use the CPU training path here. The local GGUF helper runtime may still use Vulkan acceleration through the bundled `llama.cpp` runtime when available.

For NVIDIA CUDA machines:

```powershell
uv sync --extra gpu --extra parquet
```

For a minimal no-parquet install:

```powershell
uv sync --extra cpu
```

To confirm parquet support:

```powershell
uv run python -c "import pyarrow; print(pyarrow.__version__)"
```

If `uv` is not installed yet, install it with:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Important Expectation

The launcher will not fetch packages from the network for you.

If it says a Python package is missing, install it into `.venv` locally first.

## First Launch

From the repo root, run:

```powershell
.\launch-local-builder.ps1
```

Or use:

```powershell
.\launch-local-builder.cmd
```

If you want to force a specific GGUF model at launch:

```powershell
.\launch-local-builder.ps1 -RuntimeModel ".\assistant_models\your-model.gguf"
```

If you want to control runtime device behavior:

```powershell
.\launch-local-builder.ps1 -RuntimeDeviceStrategy auto
```

Available runtime strategies:

- `auto`
  Try local GPU first, then fall back to CPU if the runtime never becomes healthy.

- `gpu`
  GPU only.

- `cpu`
  CPU only.

On Windows, the PowerShell launcher also tries to detect and load the Visual Studio x64 developer environment before starting the app.

Practical meaning:

- if Visual Studio Build Tools are installed, local CPU training has a better chance of compiling cleanly
- this is especially helpful for CPU `torch.compile` paths that expect a working C/C++ toolchain
- if Build Tools are not installed, the launcher still starts, but CPU training may be more limited or may fail later during compile steps

When the server starts, open `http://localhost:8000`.

## First 10 Minutes Troubleshooting

If the dashboard does not come up cleanly, check these first:

### 1. Missing Python packages

Symptom:

- the launcher says a package is missing
- the server exits immediately before the dashboard loads

What to check:

- `.venv` exists
- `fastapi` is installed
- `uvicorn` is installed

Rule of thumb:

- this launcher does not fetch packages for you
- install the missing packages into your local environment first, then retry

### 2. Port already in use

Symptom:

- the launcher says only one usage of the socket address is normally permitted
- the browser page stays blank or never loads

What to check:

- another copy of the builder may already be running
- another local tool may already be using the same port

Quick fix:

- close the other process using that port, or
- launch on a different port if your wrapper or host supports it

### 3. No GGUF helper model found

Symptom:

- the local runtime assistant does not start
- the dashboard shows the runtime as not ready

What to check:

- at least one real `.gguf` file exists in `assistant_models`
- the file is not empty
- the filename really ends in `.gguf`

### 4. No corpus files found

Symptom:

- builder snapshot shows zero local corpus files
- tokenizer or base training has nothing useful to read

What to check:

- your files really exist under `local_corpus`
- the files contain actual text or code
- if you are using split folders, put training data under `local_corpus/train`

### 5. Runtime falls back to CPU or fails to use GPU

Symptom:

- the builder still runs, but inference is slow
- the runtime does not stay on GPU

What to check:

- your GPU supports the local runtime path you are using
- the runtime binaries exist under `runtime/windows`
- `auto` may legitimately fall back to CPU if the GPU path never becomes healthy

### 7. CPU training fails on Windows build or compile steps

Symptom:

- tokenizer works but base training fails during a compile/build step
- logs mention MSVC, cl.exe, link.exe, or C++ build tools

What to check:

- Microsoft Visual Studio Build Tools are installed
- the Desktop development with C++ workload is installed
- you relaunched the builder after installing Build Tools

Practical note:

- the launcher will try to load the Visual Studio x64 developer environment automatically
- if the toolchain is missing, local CPU training can still fail even when Python and PyTorch are installed

### 6. Hosted inside ChattyCog but nothing appears

Symptom:

- the module registers, but the hosted dashboard does not load

What to check:

- the module folder contains `manifest.json` and `visual_load.json` at the top level
- the wrapper launch script still points at `nanochat-master/launch-local-builder.ps1`
- ChattyCog has rescanned modules after the folder was copied in

If the basic launch works but something still feels wrong, skip ahead to the full `Troubleshooting` section later in this manual.

## Recommended First-Time Workflow

If you are new, use this order:

1. Put at least one `.gguf` file in `assistant_models`.
2. Put a small but real corpus in `local_corpus`.
3. Launch the dashboard.
4. Pick the `Truth-First Teammate` preset.
5. Fill out the Assistant Studio fields.
6. Publish the design as the active identity file.
7. Ask the local runtime assistant to draft `chat_train.jsonl`.
8. Ask it to draft `chat_val.jsonl`.
9. Train the tokenizer.
10. Train a small base model first.
11. Run chat SFT.
12. Test the result in Conversation Lab.

Do not start with a giant run. Start small and prove the loop works.

## Dashboard Tour

## 1. Builder Snapshot

This is the top-level health check.

It shows:

- how many local corpus files were found
- whether the tokenizer exists
- whether the active identity file exists
- how many checkpoint families exist

If this section looks wrong, fix that first before starting jobs.

## 1.5. ECG Window

This is the small live activity card directly under the hardware summary.

Its job is simple:

- help you answer `has it crashed?`
- help you answer `is it actually doing anything?`
- help you distinguish a quiet machine from a dead one

What it tries to show:

- GPU activity first when local GPU telemetry is available
- CPU activity if GPU telemetry is not available
- a truthful builder-level fallback note if hardware telemetry is limited

How to read it:

- a moving trace with changing percentages usually means the machine is doing real work
- `Live` plus a low number can mean the builder is healthy but simply idle
- `runtime process up` means the helper runtime is still running even if there is no heavy load at that second
- if the card stays flat for a long time and the Process Log also stops changing, that is a good reason to investigate

Important caveat:

- this is a liveness window, not a deep profiler
- a low number does not always mean failure
- some jobs genuinely have quiet stretches between visible spikes

## 2. Starting Blueprints

These are presets.

Right now the important ones are:

- `Truth-First Teammate`
  Best starting point if you want correctness, explicit uncertainty, and team-member behavior.

- `Laptop Prototype`
  Smaller and cheaper for local experimentation.

Presets fill in:

- the assistant identity/design fields
- recommended tokenizer settings
- recommended base training settings
- recommended SFT settings

Presets are starting points, not guarantees.

## 3. Assistant Studio

This is where you define who your assistant is supposed to be.

Important fields:

- `Assistant Name`
- `Tone`
- `Mission`
- `Team Role`
- `Uncertainty Policy`
- `Collaboration Policy`
- `Guardrails`
- `Custom Notes`

This matters more than many new users expect.

If you want a model that says "I don't know" instead of bluffing, this is where you define that behavior clearly.

### Good Design Advice

Write these fields as operating rules, not marketing copy.

Good examples:

- "State uncertainty plainly when evidence is weak."
- "You are one member of the team, not the final authority."
- "Separate facts, assumptions, and guesses."
- "Offer verification steps when confidence is low."

Weak examples:

- "Be amazing."
- "Always be helpful."
- "Be the smartest assistant in the world."

Those are vague and do not train behavior well.

## 4. Generated Dataset Preview

When you save a design, the dashboard generates a starter identity dataset preview.

This preview is:

- local
- editable through the design fields
- intended as a seed, not a finished dataset

When you click `Publish as Active Identity File`, that identity JSONL becomes the active identity file for SFT.

## 5. Where Builder Files Live

This section shows the exact paths for:

- the cache/base directory
- tokenizer files
- active identity file
- local corpus
- assistant sandbox

If you are ever unsure where something is stored, look here first.

## 6. Stage 1: Local Corpus + Tokenizer

This stage reads only your local corpus.

Supported corpus file types include:

- `.txt`
- `.md`
- `.json`
- `.jsonl`
- `.py`
- `.js`
- `.ts`
- `.tsx`
- `.html`
- `.css`
- `.sql`
- `.yaml`
- `.yml`
- `.toml`
- `.xml`
- `.rs`
- `.go`
- `.java`
- `.cpp`
- `.c`
- `.sh`
- `.parquet` if `pyarrow` is installed locally

### What to Put in the Corpus

Put material here that you want the base model to broadly absorb:

- documents
- code
- notes
- reference text
- FAQs
- internal guides

Do not put raw chat fine-tuning files here unless you mean to use them as general corpus text.

### Main Controls

- `Corpus Directory`
- `Max Characters`
- `Document Cap`
- `Vocab Size`

### Parquet As A First-Class Input

The tokenizer and base-model corpus path can read `.parquet` alongside the text/code formats above.

Use `.parquet` when:

- you already have structured records you want to keep in a columnar format
- you want the corpus to carry metadata fields more cleanly than loose text files
- you are converting upstream data-prep outputs into a local training corpus

Requirements:

- install the local `parquet` extra with `uv sync --extra parquet`
- make sure the `.parquet` files really live under the configured corpus directory

If `pyarrow` is missing, `.parquet` files are not usable even if they are present on disk.

### Practical Advice

- start with smaller `Max Characters` if you are only testing the pipeline
- use smaller vocab sizes on weaker machines
- if the tokenizer step works, you have proven the data path is wired correctly

## 7. Stage 2: Base Model Training

This trains the underlying model from the local corpus.

Important controls:

- `Depth`
- `Head Dim`
- `Max Sequence Length`
- `Device Batch Size`
- `Total Batch Size`
- `Iterations`
- `Device Type`
- `Run Name`
- `Model Tag`

### How to Think About This Stage

- `Depth` roughly controls model size and cost
- `Device Batch Size` is often the first number to reduce if you hit memory problems
- `Iterations` controls how long the run lasts
- `Device Type` should match the hardware you really want to use

### Safe Starting Advice

If you are new, stay small:

- use the preset values first
- prefer CPU-safe or small-GPU-safe settings
- finish a short run before attempting a bigger one

### Hardware-Fit Guidance

The dashboard includes a hardware-fit pass meant to keep local runs realistic.

Use it before long runs when you are unsure about your machine.

What it is trying to do:

- detect the hardware the host actually has
- recommend safer batch, sequence, and model-shape settings
- reduce the chance of avoidable RAM or VRAM exhaustion

Even with auto-fit help, the first manual lever to lower when a run is too heavy is still usually `Device Batch Size`.

### Pause And Resume

Base training supports pause/resume through saved checkpoints.

Practical meaning:

- you can pause a long run without discarding all progress
- the next resume can continue from the latest saved checkpoint step
- checkpoint cadence matters, because resume only goes back to the most recent saved checkpoint

If you know you may need to stop and continue later, use a sensible checkpoint save interval instead of saving too rarely.

## 8. Stage 3: Chat Fine-Tuning

This is where your assistant becomes a chat assistant with the behavior you actually want.

This stage uses local JSONL conversation files only.

Typical inputs:

- `chat_train.jsonl`
- `chat_val.jsonl`
- published identity file

### Recommended Use

- publish the identity file first
- draft or review `chat_train.jsonl`
- create a smaller hold-out `chat_val.jsonl`
- include the identity file
- use a small run first

### Resume Expectations

Chat SFT is also designed around checkpoint-based continuation.

That means:

- you should keep checkpoints if you want to continue a useful run later
- resume is only as good as the latest valid saved checkpoint
- it is still worth doing a short proof run first before spending hours on a larger pass

### Why Validation Matters

If you train and validate on the same examples, you cannot tell whether the model learned the behavior or just memorized the file.

Keep some examples for validation only.

## 9. Stage 4: Benchmark + Smoke Test

This is a fast local sanity check.

It is meant for iteration and for keeping a repeatable local benchmark config from run to run.

Use it to:

- verify the checkpoint loads
- get quick BPB feedback on your local corpus
- sample outputs and see whether the run is obviously broken
- run a stable local benchmark for easier comparison across runs
- auto-tune the next run settings from prior outcomes

## 10. Stage 5: Chat RL

This stage extends the chat pipeline beyond supervised fine-tuning.

Use it only after Chat SFT is already working and you have an SFT checkpoint available to load.

Practical meaning:

- SFT teaches the assistant the basic conversation format and behavior
- Chat RL is where you experiment with pushing that behavior further

Recommended order:

1. complete Stage 3 Chat SFT
2. confirm the SFT checkpoint can load
3. run Chat RL only after the supervised baseline looks sane

If this stage fails immediately, the most likely cause is that the requested SFT checkpoint does not exist yet.

RL continuation also depends on valid saved checkpoints, so treat checkpoint retention as part of the workflow rather than an optional extra if you expect to pause and return later.

## 11. Stage 6: Chat Eval

This is a dedicated evaluation stage for the chat-model side of the pipeline.

Use it when you want a cleaner read on the current SFT or RL-derived chat checkpoint.

It is useful for:

- checking whether a chat checkpoint loads correctly
- comparing chat-stage runs more consistently
- getting a repeatable read before or after RL experiments

If this stage fails immediately, the most likely cause is that no SFT checkpoint is available yet.

## 12. Managed llama.cpp Server

This runs your local GGUF assistant.

It uses the bundled Windows runtime from `runtime/windows`.

The runtime manager:

- discovers GGUF files under `assistant_models`, `models`, and `runtime/models`
- prefers a recommended default model if you do not pick one
- tries GPU first in `auto` mode
- falls back to CPU if GPU startup does not become healthy

### Main Runtime Controls

- `Model Path (.gguf)`
- `Device Strategy`
- `Preferred Device`
- `Port`
- `Context Size`
- `Threads`
- `HTTP Threads`
- `Parallel Slots`
- `GPU Layers`

### When to Use the Local Runtime Assistant

Use it when you want help with:

- reviewing the current builder state
- drafting SFT data
- checking what happened in recent jobs
- reading or editing sandbox files
- launching supported jobs from chat

## 13. Conversation Lab

This is the main work area for talking to the local assistant.

Every local-runtime request includes:

- the current builder summary
- a recent tail of the activity log
- any sandbox or corpus file you explicitly include
- a built-in cockpit protocol that reminds the GGUF assistant it is inside llm-tweaker, that it should act as an LLM-building tutor, and that it should return direct answers instead of reasoning-only output

If a model ignores that protocol and emits reasoning-only text anyway, the dashboard will warn you. That is usually a model/prompt-fit issue, not a dashboard crash.

If `Assistant Actions` is enabled, the local assistant can also use local tools.

### What the Assistant Can Do

With assistant actions enabled, it can:

- inspect builder state
- inspect recent activity
- list corpus files
- read corpus files
- write corpus files
- delete corpus files
- copy reviewed sandbox files into the corpus
- list sandbox files
- read sandbox files
- write sandbox files
- delete sandbox files
- draft validated SFT JSONL data
- list jobs
- inspect job status
- stop jobs
- launch supported local jobs

### What It Cannot Do Through the Dashboard Tools

- write outside `assistant_sandbox` and `local_corpus`
- browse the whole machine through the dashboard tool layer
- silently change arbitrary files in the repo

That workspace boundary is intentional.

### Good Prompts for the Local Assistant

- "Review the current builder state and tell me the next step."
- "Draft 20 training examples into `chat_train.jsonl` that teach explicit uncertainty."
- "Create `chat_val.jsonl` with 8 hold-out examples that test team-member behavior."
- "Read the selected sandbox or corpus file and tell me what is weak or repetitive."
- "Launch chat SFT using the current training files."
- "Check recent activity and summarize what the last job did."

## 14. assistant_sandbox Files

This is the assistant's working desk.

Use it for:

- chat training files
- validation files
- notes
- prompt drafts
- ideas to review later

The assistant only sees sandbox files you explicitly include in chat.

That means:

- you stay in control of context
- the assistant does not automatically read every file in the sandbox
- you decide what each chat turn is allowed to see

## 15. Sandbox Editor

You can create and edit sandbox files directly in the dashboard.

Useful patterns:

- write or paste a first draft yourself
- ask the assistant to improve it
- save the assistant's last reply into a sandbox file
- review and clean the data before training

This is a good place to keep:

- `chat_train.jsonl`
- `chat_val.jsonl`
- scratch notes
- evaluation ideas
- future training plans

## 16. Process Log

The process log is one of the most important features for debugging.

It records local events such as:

- job start and stop
- job output
- sandbox reads and writes
- runtime start and stop
- runtime errors
- chat turns
- assistant tool actions

The local assistant receives a compact tail of this log on each local-runtime request.

This is how it can "see" what has been happening recently.

The underlying log file is:

`builder_logs/activity.jsonl`

## 17. Common Functions In Plain Language

This section is a practical glossary for the main buttons, actions, and functions in the dashboard.

Use it when you are looking at a control and thinking:

- what is this for?
- when should I use it?
- what happens if I click it?

### Quick Reference Table

| Function | Plain-language purpose | Use it when |
|----------|------------------------|-------------|
| `Fit To Hardware` | Fill forms with safer machine-sized starter settings | You want a safer default before running jobs |
| `Save Design` | Save the current assistant blueprint draft | You want to keep or revisit the current design |
| `Publish As Active Identity File` | Turn the current design into the active identity training file | You want SFT to use this identity |
| `Draft With GGUF` | Ask the helper model to draft a blueprint from a plain-language goal | You want a first draft without filling every field manually |
| `Train Tokenizer` | Build the tokenizer from the local corpus | Your corpus is ready and you want to begin the pipeline |
| `Run Tokenizer Eval` | Check tokenizer quality on sample corpus data | You want a quick tokenizer sanity check |
| `Launch Base Training` | Train the underlying language model from the corpus | The tokenizer exists and you want a base model |
| `Run Base Eval` | Smoke-test the current base checkpoint | You want a quick quality check on the base model |
| `Launch Chat SFT` | Fine-tune the model on local chat JSONL files | You have train/val chat files and want assistant behavior |
| `Run Benchmark` | Run a repeatable local comparison pass | You want to compare runs more consistently |
| `Auto Tune Next Run` | Suggest next-run settings from recent benchmark history | You want a guided next step instead of guessing |
| `Launch Chat RL` | Run advanced RL-style tuning on top of SFT | SFT already works and you are experimenting further |
| `Run Chat Eval` | Evaluate the chat-stage checkpoint | You want a cleaner read on SFT or RL behavior |
| `Pause Job` | Stop a running resumable job at a safe point | You need to stop without losing all progress |
| `Resume Job` | Continue from the latest saved checkpoint | You paused a base, SFT, or RL job earlier |
| `Copy Resume To Form` | Prefill the matching form with resume settings | You want to inspect or adjust resume values manually |
| `Use Checkpoint In Eval/Load` | Copy checkpoint details into eval/load controls | You want to test a specific run's checkpoint |
| `Load Chat Model` | Load an internal nanochat checkpoint into the chat bench | You want to test your trained model directly |
| `Start Local Runtime` | Start the bundled GGUF helper runtime | You want the helper assistant available |
| `Stop Local Runtime` | Stop the helper runtime process | You want to free resources or restart cleanly |
| `Apply Cockpit To Bench` | Sync cockpit settings into the testing surfaces | You changed backend or sampling controls |
| `Activate Backend Profile` | Switch the active backend/model lane | You want to route testing to a different model source |
| `Reset Saved Controls` | Clear saved preferences and restore sane defaults | The dashboard controls feel messy or confusing |
| `Run Quick Prompt` | Send one prompt and get one reply | You want the fastest smoke test |
| `Copy To Conversation Lab` | Move a quick prompt into the longer chat area | A one-shot test is worth expanding |
| `Send In Conversation Lab` | Send a full chat turn to the selected provider | You want help, drafting, or deeper testing |
| `Save File / Load File / Delete File` | Manage sandbox or corpus files in place | You want to work with local files from the dashboard |
| `Save Last Assistant Reply` | Turn the last assistant answer into a sandbox file | The reply is worth keeping or editing |
| `Copy Sandbox To Corpus` | Move a reviewed draft from sandbox into corpus | A draft is ready to become training data |

### Fit To Hardware

This function checks the machine you are running on and fills the forms with safer starter values.

Use it when:

- you are new
- you changed machines
- you are unsure whether the current settings are too heavy

Plain English:

- it is the dashboard's `please do not let me accidentally bully my hardware` button

### Save Design

This saves the current assistant identity blueprint without making it the active published identity file yet.

Use it when:

- you want to keep a draft
- you want to come back later
- you want the design tray to remember this version

Plain English:

- it saves the assistant idea
- it does not yet make that idea the active training identity

### Publish As Active Identity File

This takes the current design and writes the generated identity conversations into the active identity JSONL file used by the training flow.

Use it when:

- you are happy with the current assistant design
- you want Stage 3 Chat SFT to use that identity

Plain English:

- this is the step that turns the design into a real file the trainer can use

### Draft With GGUF

This asks the helper GGUF model to draft an assistant design and starter settings from a plain-language goal.

Use it when:

- you know the outcome you want
- you do not want to write every identity field from scratch

Plain English:

- you describe the assistant in normal language
- the helper drafts a first version for you to review and edit

### Train Tokenizer

This starts tokenizer training from the local corpus.

Use it when:

- you have real corpus files in place
- you want to begin the actual training pipeline

Plain English:

- it teaches the system how to break your text into tokens before model training happens

### Run Tokenizer Eval

This checks the tokenizer quality on a sample of your corpus.

Use it when:

- you want a quick compression sanity check
- you are comparing tokenizer settings

Plain English:

- it tells you whether the tokenizer looks reasonable before you invest more work downstream

### Launch Base Training

This starts training the underlying language model from the local corpus.

Use it when:

- the tokenizer is ready
- you want to build the base model stage

Plain English:

- this is the step where the model starts learning from the general reading pile

### Run Base Eval

This runs a quick evaluation and sampling pass against the current base checkpoint.

Use it when:

- you want to see whether the base run is obviously broken
- you want a repeatable local quality check

Plain English:

- it is a smoke test for the base model

### Launch Chat SFT

This starts supervised fine-tuning on your local chat JSONL files.

Use it when:

- the base stage exists
- you have `chat_train.jsonl` and `chat_val.jsonl`
- you want the model to behave like an assistant instead of just a base language model

Plain English:

- this is where the model learns the chat behavior and identity you actually care about

### Run Benchmark

This runs the saved local benchmark flow so you can compare runs more consistently.

Use it when:

- you want a more stable comparison than casual chatting
- you are iterating and need a repeatable reference point

Plain English:

- it helps you compare runs with less guesswork

### Auto Tune Next Run

This looks at recent benchmark history and suggests settings for the next run.

Use it when:

- you already have a few runs behind you
- you want help nudging settings instead of guessing from zero

Plain English:

- it is the dashboard's `based on what we already saw, try this next` helper

### Launch Chat RL

This starts reinforcement-learning-style chat tuning on top of an SFT checkpoint.

Use it when:

- SFT is already working
- you are deliberately experimenting beyond the supervised baseline

Plain English:

- this is the more advanced tuning lane, not the beginner starting point

### Run Chat Eval

This evaluates the chat-stage model against task-style checks.

Use it when:

- you want a cleaner read on the chat checkpoint
- you want to compare SFT and RL outputs more consistently

Plain English:

- it is the report card for the chat model stage

### Pause Job

This asks a running train job to stop at the next safe point so it can later continue from the latest saved checkpoint.

Use it when:

- you need the machine back
- the run is taking longer than expected
- you want to stop without throwing away all progress

Plain English:

- it means `stop as safely as you can and leave me somewhere resumable`

### Resume Job

This launches a continuation run from the latest saved checkpoint for a paused or stopped resumable job.

Use it when:

- you previously paused a base, SFT, or RL run
- a checkpoint exists

Plain English:

- it picks the run back up from where the last usable checkpoint left off

### Copy Resume To Form

This copies resume-related values from a selected job back into the matching form.

Use it when:

- you want to inspect or adjust the resume settings manually before relaunching

Plain English:

- it pre-fills the form so you do not have to reconstruct resume settings by hand

### Use Checkpoint In Eval/Load

This copies the selected job's checkpoint details into the evaluation or model-load controls.

Use it when:

- you want to test a specific checkpoint
- you do not want to type the tag and step manually

Plain English:

- it is a shortcut for `use this run's checkpoint over there`

### Load Chat Model

This loads an internal nanochat SFT or RL checkpoint into the chat runtime for direct testing.

Use it when:

- you want to test the trained internal model instead of only the helper GGUF

Plain English:

- it puts your own trained model into the chat bench

### Start Local Runtime

This starts the bundled local `llama.cpp` server for the GGUF helper assistant.

Use it when:

- you want the helper assistant available
- you want to use Quick Prompt or Conversation Lab against the GGUF path

Plain English:

- it turns on the local helper model service

### Stop Local Runtime

This stops the helper GGUF runtime process.

Use it when:

- you want to free hardware resources
- the runtime is misbehaving
- you want to restart cleanly

Plain English:

- it shuts the helper assistant service down

### Apply Cockpit To Bench

This copies the cockpit sampling and backend settings into the quick prompt and conversation bench controls.

Use it when:

- you changed temperature, max tokens, backend profile, or action mode in the cockpit
- you want the testing tools to use those same settings

Plain English:

- it syncs the operator controls into the places where you actually chat

### Activate Backend Profile

This switches the active helper/testing backend profile according to the cockpit selection.

Use it when:

- you want to route tests through local runtime, current internal chat, latest SFT, or latest RL

Plain English:

- it tells the dashboard which model lane you want to use right now

### Reset Saved Controls

This clears stored cockpit and runtime preferences and restores safer defaults.

Use it when:

- the controls feel messy
- an old experiment left confusing saved settings behind

Plain English:

- it is the `put the dashboard back into a sane state` button

### Run Quick Prompt

This sends one prompt and gets one reply back without building up a conversation history.

Use it when:

- you want a fast smoke test
- you want to compare a single answer before doing a longer chat

Plain English:

- it is the fastest way to ask `does this model answer at all, and does it sound sane?`

### Copy To Conversation Lab

This moves the current quick prompt content into the longer chat area.

Use it when:

- a one-shot test is worth expanding into a fuller conversation

Plain English:

- it promotes a quick experiment into a longer test

### Send In Conversation Lab

This sends your message to the selected chat provider with the current assistant settings and optional file context.

Use it when:

- you want the helper assistant to review state, explain tradeoffs, or draft local files

Plain English:

- this is the main `talk to the builder assistant` function

### Save File / Load File / Delete File

These functions in the sandbox and corpus editors do exactly what they sound like:

- `Save File` writes the current editor content
- `Load File` reads the selected file back into the editor
- `Delete File` removes the selected file

Use them when:

- you want to manage local working files without leaving the dashboard

Plain English:

- they are the file-management buttons for the two local workspaces

### Save Last Assistant Reply

This saves the assistant's most recent reply into a sandbox file.

Use it when:

- the assistant drafted something worth keeping
- you want to turn a reply into a file for later editing

Plain English:

- it captures a useful answer before it scrolls away

### Copy Sandbox To Corpus

This copies a reviewed sandbox file into the local corpus area.

Use it when:

- you drafted something in the sandbox and now want it to become part of the corpus

Plain English:

- it is the bridge from `draft workspace` to `training data workspace`

## Understanding the Training Data Types

You will work with three different kinds of data here.

## 1. Local Corpus Data

Used for:

- tokenizer training
- base model training

This is broad material.

Think of it as the general reading pile.

## 2. Identity Data

Used for:

- teaching the model how it should act

This comes from the Assistant Studio design and publish flow.

Think of it as the assistant charter.

## 3. Chat SFT Data

Used for:

- teaching concrete assistant behavior in conversation

This usually lives in:

- `assistant_sandbox/chat_train.jsonl`
- `assistant_sandbox/chat_val.jsonl`

Think of it as example conversations that show the model how to behave.

## SFT JSONL Format

Each line must be one full conversation.

Each conversation is a JSON array of alternating message objects.

Each conversation must:

- start with a `user` message
- alternate `user`, `assistant`, `user`, `assistant`
- contain string `role` and `content` fields

Example:

```json
[{"role":"user","content":"What should you do when you are unsure?"},{"role":"assistant","content":"Say that clearly, separate facts from guesses, and explain how to verify the answer."}]
[{"role":"user","content":"Are you the whole team?"},{"role":"assistant","content":"No. I am one member of the team and I should make handoffs explicit when another person or tool should own the next step."}]
```

The assistant's `draft_sft_data` action writes this format for you.

## Recommended Behavior Patterns to Train

If your goal is correctness over speed, teach that directly.

Good example themes:

- admitting uncertainty
- separating facts from guesses
- asking for missing constraints
- showing tradeoffs
- offering verification steps
- making handoffs explicit
- acting like a teammate, not a sole authority

Avoid overloading the dataset with only slogans.

A good file has examples that force the behavior to appear in context.

## Suggested Workflow for Better Behavior

Use this pattern:

1. Define the design clearly in Assistant Studio.
2. Publish the identity file.
3. Draft 20 to 100 focused SFT conversations, not thousands of weak ones.
4. Keep a smaller validation file aside.
5. Fine-tune.
6. Test in Conversation Lab.
7. Add more examples only for the behaviors that still fail.

This usually beats dumping huge amounts of vague synthetic data into the model.

## Practical Suggestions

### Start Small

Your first goal is not "best model".

Your first goal is:

**prove the loop works end to end**

That means:

- the runtime starts
- the corpus is discovered
- the tokenizer trains
- the base model trains
- SFT runs
- the result can be tested

Once that works, then optimize.

### Keep Validation Separate

Always reserve some examples for `chat_val.jsonl`.

If the model only looks good on the same file it trained on, that is weak evidence.

### Use the Assistant as a Drafting Partner

The local assistant is best used for:

- first drafts
- consistency checks
- next-step suggestions
- log review

It is not a substitute for reviewing the data before training.

### Prefer Concrete Examples Over Abstract Rules

A single good conversation example is often worth more than ten vague instructions.

Bad:

- "Be honest."

Better:

- user asks for a fact with missing evidence
- assistant says it is unsure
- assistant explains what is known
- assistant suggests what to verify next

### Keep the Team Role Explicit

If you want the model to behave like one teammate among many, say that repeatedly in:

- Assistant Studio
- identity data
- SFT data
- validation data

Do not assume the model will infer it from one sentence.

### Watch the Process Log

If something feels confusing, read the log before changing ten settings at once.

The log often tells you:

- whether a job actually started
- whether a file was written
- which model was loaded
- whether the runtime fell back to CPU

## Common Tasks

## Ask the Assistant to Draft Training Data

Example prompt:

`Draft 12 chat training examples into chat_train.jsonl that teach you to say "I don't know" when confidence is low, separate facts from guesses, and behave like one member of the user's team.`

## Ask the Assistant to Draft Validation Data

Example prompt:

`Create chat_val.jsonl with 8 hold-out examples that test uncertainty, handoffs, and asking for missing constraints.`

## Ask the Assistant to Review a File

Example prompt:

`Read the selected sandbox or corpus file and tell me which examples are weak, repetitive, or likely to teach overconfidence.`

## Ask the Assistant to Review the Current State

Example prompt:

`Review the current builder state and process log, then tell me the next two steps and the main risk right now.`

## Ask the Assistant to Launch a Job

Example prompt:

`Launch chat SFT with the current training and validation files.`

## Troubleshooting

## The Dashboard Does Not Start

Check:

- `.venv` exists
- `fastapi` exists in `.venv`
- `uvicorn` exists in `.venv`

The launcher does not auto-install missing packages.

## The Local Runtime Does Not Start

Check:

- a real `.gguf` file exists in `assistant_models`
- `runtime/windows/llama-server.exe` exists
- the selected port is free
- your device strategy matches reality

If `auto` fails on GPU, the runtime should fall back to CPU.

If `gpu` is forced and no usable device exists, startup will fail.

## The ECG Window Looks Flat Or Quiet

Check:

- whether the Process Log is still moving
- whether a job is actually running
- whether the local runtime is still marked ready

Interpret it like this:

- flat trace plus changing logs usually means the current work is light or bursty, not necessarily broken
- flat trace plus no log movement for a long time is a better reason to suspect a stall
- CPU-only systems may show CPU activity instead of GPU activity, which is expected

If you want more confidence, compare three things together:

- ECG window
- Job Queue status
- Process Log tail

## No Models Appear in the Runtime Model List

Check:

- the files are real GGUFs and not empty placeholders
- they are under `assistant_models`, `models`, or `runtime/models`

## The Corpus Count Is Zero

Check:

- files really exist under `local_corpus`
- the file types are supported
- if you are using split folders, use `local_corpus/train`

## Parquet Files Do Not Work

You need local `pyarrow` support for `.parquet`.

Preferred install command from `nanochat-master/`:

```powershell
uv sync --extra parquet
```

Manual pip fallback:

```powershell
python -m pip install pyarrow
```

For local Windows CPU training, you may also need Microsoft Visual Studio Build Tools with the Desktop C++ workload so compile-based training paths can run correctly.

If you do not want that dependency, convert the data to `.txt`, `.json`, or `.jsonl`.

## Training Is Too Slow

Reduce:

- model depth
- device batch size
- total batch size
- sequence length
- iteration count

Also prefer the `Laptop Prototype` preset if you are only validating the pipeline.

## You Hit Memory Errors

The first number to lower is usually `Device Batch Size`.

After that, reduce:

- `Depth`
- `Max Sequence Length`
- `Total Batch Size`

## The Assistant Gives Weak Drafts

Usually one of these is true:

- the system prompt is too vague
- the selected sandbox or corpus file gives poor context
- the design is vague
- your request is too broad

Fix it by being more concrete.

Bad prompt:

- "Make this better."

Better prompt:

- "Draft 10 short training examples that teach explicit uncertainty, one-team-member framing, and clean handoffs."

## The Model Still Hallucinates

That is normal after an early draft.

Typical fixes:

- add more examples where the correct behavior is to admit uncertainty
- add examples that separate facts from assumptions
- add validation cases that test bluffing behavior
- keep the identity data aligned with the same policy
- retrain and test again

## Suggested Operating Style

If you want the system to become a correctness-first assistant, keep reinforcing the same pattern:

- clear mission
- clear uncertainty policy
- clear team role
- specific conversation examples
- repeated testing
- corrections when behavior drifts

The best results usually come from consistent, boring clarity rather than clever wording.

## Final Advice

Treat this system like a local workshop:

- the GGUF assistant helps you operate the bench
- the sandbox holds your drafts
- the process log tells you what happened
- the training stages turn local data into a model

If you are unsure what to do next, start with the simplest loop:

1. Launch the dashboard.
2. Pick `Truth-First Teammate`.
3. Publish the identity file.
4. Ask the assistant to draft `chat_train.jsonl`.
5. Review it.
6. Run a small tokenizer job.
7. Run a small base training job.
8. Run chat SFT.
9. Test the result.

That loop is the core of the system.
