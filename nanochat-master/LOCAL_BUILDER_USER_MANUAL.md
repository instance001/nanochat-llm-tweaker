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
- `pyarrow` only if you want to use `.parquet` files in the corpus

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

## 10. Managed llama.cpp Server

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

## 11. Conversation Lab

This is the main work area for talking to the local assistant.

Every local-runtime request includes:

- the current builder summary
- a recent tail of the activity log
- any sandbox or corpus file you explicitly include

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

## 12. assistant_sandbox Files

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

## 13. Sandbox Editor

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

## 14. Process Log

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
