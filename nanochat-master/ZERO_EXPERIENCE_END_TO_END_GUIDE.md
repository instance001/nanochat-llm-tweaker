# Zero-Experience End-to-End Guide

## Who This Is For

This guide is for someone who is completely new to:

- local LLM tools
- Python environments
- GGUF models
- tokenizer training
- base-model training
- SFT JSONL files
- running local web dashboards

If that is you, this is the right document.

This guide is intentionally practical.

It explains:

- what each major part of the builder does
- what files you need before you can use it
- terminal commands to set up the Python side
- what to search for when you do not have the files yet
- what each dashboard area is for
- what each stage expects as input
- what outputs you should expect
- what to do when something fails

## What This Builder Actually Is

This project is a local-only LLM builder based on `nanochat`, but adapted into a guided dashboard.

There are two different model lanes here:

1. The **helper assistant**
   This is a local `.gguf` model you place into `assistant_models/`.
   It helps you operate the builder through the dashboard.

2. The **model you train**
   This is the internal `nanochat` model you create through tokenizer training, base-model training, and chat fine-tuning.

Plain English version:

- the GGUF helps you build
- the internal model is what you are actually training

## Before You Touch The Dashboard

You need four categories of things:

1. A Windows machine
2. Python installed
3. A helper `.gguf` model
4. Some real local training data

If you want to do local Windows CPU training, add a fifth practical requirement:

5. Microsoft Visual Studio Build Tools with the Desktop C++ workload

You do not need:

- cloud inference
- remote datasets
- a paid API
- a giant GPU cluster

## Folder Landmarks

These are the folders you will use most:

- `assistant_models/`
  Put helper `.gguf` models here.

- `local_corpus/`
  Put your local training corpus here.
  This can include `.txt`, `.md`, `.json`, `.jsonl`, code files, and `.parquet`.

- `assistant_sandbox/`
  Use this for drafts, notes, `chat_train.jsonl`, and `chat_val.jsonl`.

- `builder_logs/`
  Activity logs and benchmark history go here.

- `runtime/windows/`
  Bundled local `llama.cpp` runtime files.

## Step 1: Install Python

If you do not already have Python 3.10+ installed:

1. Search Google for:
   `Python 3.12 Windows download`

2. Install Python from the official Python site.

3. During install, enable:
   `Add Python to PATH`

4. Open PowerShell and verify:

```powershell
python --version
```

If that works, Python is available.

If it says Python is not found, search Google for:

- `Windows python add to path`
- `python command not recognized windows`

## Step 2: Install Basic Python Packages

This builder will create a local `.venv`, but it does not automatically download missing packages for you.

Open PowerShell in `nanochat-master/` and run:

```powershell
python -m venv .venv --system-site-packages
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install fastapi uvicorn psutil python-dotenv regex rustbpe scipy setuptools tabulate tiktoken tokenizers transformers zstandard datasets matplotlib ipykernel kernels
```

If you want local `.parquet` corpus support, also install:

```powershell
python -m pip install pyarrow
```

If you want local Windows CPU training to work reliably, install Microsoft Visual Studio Build Tools too.

Google search terms:

- `Visual Studio Build Tools download`
- `Desktop development with C++ workload`

What to install:

- Microsoft Visual Studio Build Tools
- the `Desktop development with C++` workload

Why this matters:

- some local CPU training paths rely on a working Windows C/C++ toolchain
- without it, training can fail later even if Python packages installed correctly

If you want to run training inside this same environment, install PyTorch too.

CPU-only example:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

If you have an NVIDIA GPU and want GPU training, search Google for:

- `PyTorch install CUDA Windows`
- `PyTorch previous versions install selector`

Then install the correct wheel for your CUDA setup.

If package installation fails, useful Google searches are:

- `pip install fastapi windows failed`
- `pip install torch windows cpu`
- `pip install pyarrow windows`
- `PowerShell execution policy Activate.ps1`

If PowerShell blocks environment activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then try again.

## Step 3: Get A Helper GGUF Model

The dashboard assistant needs at least one `.gguf` file.

Put it in:

- `assistant_models/`

Good search terms:

- `GGUF instruct model download`
- `Qwen GGUF instruct Hugging Face`
- `Mistral 7B Instruct GGUF`
- `small GGUF instruct model laptop`

What you are looking for:

- a real `.gguf` file
- usually an instruct or chat model
- ideally not enormous for your machine

If you are new, start with a smaller instruct model before trying very large ones.

When you download one, put it in:

```text
nanochat-master\assistant_models\
```

## Step 4: Get Local Corpus Data

The base model and tokenizer need some real corpus data.

Put it in:

```text
nanochat-master\local_corpus\
```

Or split it like this:

```text
nanochat-master\local_corpus\train\
nanochat-master\local_corpus\val\
```

Examples of acceptable corpus content:

- notes
- docs
- markdown files
- code files
- FAQs
- internal manuals
- `.json` or `.jsonl`
- `.parquet`

What not to do:

- do not leave `local_corpus/` empty
- do not expect the builder to download a corpus for you
- do not start with millions of files if you are just proving the loop works

Beginner-friendly search ideas if you are not sure what a corpus should look like:

- `what is training corpus for llm`
- `sample technical writing corpus text files`
- `how to prepare local corpus for language model`
- `jsonl vs parquet machine learning dataset`

## Step 5: Launch The Builder

From inside `nanochat-master/`, run:

```powershell
.\launch-local-builder.ps1
```

Or:

```powershell
.\launch-local-builder.cmd
```

If you want a specific GGUF:

```powershell
.\launch-local-builder.ps1 -RuntimeModel ".\assistant_models\your-model.gguf"
```

Helpful note:

- on Windows, the launcher will try to detect and load the Visual Studio x64 developer environment automatically if Build Tools are installed
- this improves the odds that local CPU training runs can compile and start cleanly

If the launcher fails, check:

1. Python exists
2. `.venv` exists
3. `fastapi` is installed
4. `uvicorn` is installed

If the browser does not open, go to:

- `http://localhost:8000`

If the ChattyCog wrapper is being used instead, the outer module may host it on:

- `http://127.0.0.1:8765`

## Step 6: What Each Dashboard Area Does

This section is the beginner map of the GUI.

### Builder Snapshot

Purpose:

- quick health check

What it tells you:

- how many corpus files were found
- whether the tokenizer exists
- whether the identity file exists
- whether checkpoints exist
- whether the local runtime is ready

What to do here:

- if this looks wrong, fix the missing inputs before launching jobs

### ECG Window

Purpose:

- answer `has it crashed?`
- answer `is the machine still doing work?`

What it shows:

- a small live activity trace
- a current percentage
- whether the signal is coming from GPU, CPU, or a builder-level fallback
- a short note such as `runtime process up` or how long ago the last builder event happened

How to read it:

- changing percentages and a moving trace usually mean the builder is alive
- a low percentage does not automatically mean a bug
- if the trace is flat and the Process Log also stops changing for a long time, start investigating

Beginner translation:

- moving line = probably working
- quiet line + quiet logs = check what is going on

### Starting Blueprints

Purpose:

- presets for beginners

What to do:

- start with `Truth-First Teammate`

Why:

- it fills in a safer default assistant design
- it keeps the project focused on correctness and explicit uncertainty

### Assistant Studio

Purpose:

- define your assistant behavior

Important fields:

- Assistant Name
- Tone
- Mission
- Team Role
- Uncertainty Policy
- Collaboration Policy
- Guardrails
- Custom Notes

What to do:

1. keep the instructions concrete
2. write behavior rules, not marketing language
3. save the design

Good examples:

- `Say when confidence is low.`
- `Separate facts from guesses.`
- `Act like one teammate, not the whole team.`

Weak examples:

- `Be smart.`
- `Be amazing.`

### Generated Dataset Preview

Purpose:

- shows the starter identity dataset produced from Assistant Studio

What to do:

1. read it
2. make sure it matches the assistant you want
3. click `Publish as Active Identity File`

Output:

- an active identity JSONL file used later in SFT

### Where Builder Files Live

Purpose:

- shows the actual paths being used

Use this when:

- you are unsure where tokenizer files were written
- you want to inspect the identity file
- you want to confirm the local corpus path

### Stage 1: Local Corpus + Tokenizer

Purpose:

- build a tokenizer from your local corpus

Inputs:

- files in `local_corpus/`

Main fields:

- Corpus Directory
- Max Characters
- Document Cap
- Vocab Size

What the buttons do:

- `Train Tokenizer`
  Builds tokenizer files from the local corpus.

- `Run Tokenizer Eval`
  Gives you a local compression sanity check.

What you should do as a beginner:

1. keep the corpus small at first
2. leave the preset values alone unless you have a reason
3. run tokenizer training
4. confirm the tokenizer now appears as ready

Parquet note:

- `.parquet` is supported here too, not just `.txt`
- install `pyarrow` first or those files will not be usable
- if you are brand new, it is still fine to prove the flow with a few `.txt` files first

What output you should expect:

- tokenizer files under the builder cache directory

If this fails, search:

- `tokenizer training failed python`
- `rustbpe install windows`
- `no local documents found train split`

### Stage 2: Base Model Training

Purpose:

- train the underlying language model from the local corpus

Inputs:

- tokenizer from Stage 1
- local corpus

Main fields:

- Depth
- Head Dim
- Max Sequence Length
- Device Batch Size
- Total Batch Size
- Iterations
- Device Type
- Run Name
- Model Tag

What the buttons do:

- `Launch Base Training`
  Starts base-model training as a background job.

- `Run Base Eval`
  Runs quick evaluation for the selected checkpoint.

- `Fit To Hardware`
  Detects the host machine and recommends safer settings for the next run.

Beginner advice:

1. stay small
2. use CPU if you are unsure
3. reduce `Device Batch Size` first if memory fails
4. finish one short run before chasing quality

Why `Fit To Hardware` matters:

- local machines vary wildly
- a workstation GPU, a laptop GPU, and CPU-only hardware should not start from the same assumptions
- the fit pass is there to reduce the chance of out-of-memory failures or punishing consumer hardware unnecessarily

Pause and resume:

- base training can be paused
- later you can resume from the latest saved checkpoint instead of starting over
- this only works as well as your checkpoint save frequency, so do not save too rarely on long jobs

What output you should expect:

- base checkpoints
- job logs
- validation BPB values if evaluation is enabled

If this fails, search:

- `pytorch out of memory reduce batch size`
- `torch compile windows issue`
- `bf16 not supported windows pytorch`

### Stage 3: Chat Fine-Tuning

Purpose:

- teach the base model how to behave in conversation

Inputs:

- base model checkpoint
- published identity file
- `chat_train.jsonl`
- `chat_val.jsonl`

Where these files usually live:

- `assistant_sandbox/chat_train.jsonl`
- `assistant_sandbox/chat_val.jsonl`

What the button does:

- `Launch Chat SFT`
  Starts supervised fine-tuning on local conversation files.

Beginner advice:

1. publish the identity file first
2. draft a small `chat_train.jsonl`
3. create a separate `chat_val.jsonl`
4. do not train and validate on the same exact examples

Pause and resume also matter here:

- keep checkpoints if the run may need to continue later
- resume continues from a saved checkpoint, not from every in-between minibatch

What good training examples should teach:

- explicit uncertainty
- clean handoffs
- asking for missing constraints
- behaving like a teammate

If you do not know the format yet, search:

- `jsonl conversation dataset llm`
- `supervised fine tuning jsonl chat format`

### Stage 4: Benchmark + Smoke Test

Purpose:

- quick repeatable evaluation

What it checks:

- BPB on your corpus
- checkpoint loading
- sample outputs

What the buttons do:

- `Run Base Eval`
  Evaluate a chosen checkpoint.

- `Run Benchmark`
  Run a stable local benchmark config.

- `Auto Tune Next Run`
  Recommend next settings from prior results.

Beginner advice:

- use this after every meaningful training run

### Stage 5: Chat RL

Purpose:

- continue improving the chat model after SFT using a reinforcement-learning style stage

What it needs before it can work:

- a successful base model run
- a successful Chat SFT run
- an SFT checkpoint to load

What the button does:

- `Launch Chat RL`
  Starts the local RL job for the selected or latest SFT checkpoint.

When a beginner should use it:

1. only after you already have a working SFT model
2. only after you can load that SFT model successfully
3. only after you have a clear reason to push behavior further

What this stage is good for:

- behavior shaping after basic chat format is already learned
- tightening preference behavior
- experimenting after the supervised path is already stable

Like the earlier stages, continuing later depends on saved checkpoints. If you expect to stop and come back, keep checkpoint saving enabled.

If this fails, the most common reason is:

- no SFT checkpoint exists yet

If this fails, search:

- `llm chat rl basics`
- `sft checkpoint not found`
- `reinforcement learning fine tuning local llm`

### Stage 6: Chat Eval

Purpose:

- run a dedicated chat-model evaluation pass against an SFT checkpoint

What it needs before it can work:

- a successful Chat SFT run
- a valid SFT checkpoint

What the button does:

- `Run Chat Eval`
  Evaluates the selected or latest chat checkpoint.

When to use it:

1. after Chat SFT
2. after Chat RL if you want a quick follow-up check
3. whenever you want a repeatable chat-stage comparison point

What to expect:

- logs showing whether the checkpoint loaded correctly
- evaluation output for the current chat model family

If this fails, the most common reason is:

- no SFT checkpoint exists yet

If this fails, search:

- `chat model evaluation local llm`
- `no sft checkpoint found`
- `python checkpoint path not found`

### Internal Chat Runtime

Purpose:

- load an internally trained chat model

What the buttons do:

- `Load Chat Model`
  Load a selected internal chat checkpoint for chatting.

- `Load Latest SFT`
  Load the latest SFT checkpoint automatically.

Use this after:

- base training plus SFT have already succeeded
- optionally after Chat RL if you want to test the newest chat-stage checkpoint

### Managed llama.cpp Server

Purpose:

- start the helper GGUF model

Main fields:

- Model Path
- Device Strategy
- Preferred Device
- Port
- Context Size
- Threads
- HTTP Threads
- Parallel Slots
- GPU Layers

What the buttons do:

- `Start Local Runtime`
  Starts the helper `.gguf` assistant through bundled `llama.cpp`.

- `Stop Local Runtime`
  Stops it.

Beginner advice:

- use `auto` first
- if startup is unstable, try `cpu`

If the runtime does not start, search:

- `gguf model not detected`
- `llama.cpp windows vulkan failed`
- `llama-server.exe runtime windows`

If you are unsure whether the builder is active or stalled after startup, use the ECG window together with the Process Log:

- the ECG window gives you a quick liveness signal
- the Process Log tells you what actually happened
- use both before assuming the app is frozen

### Conversation Lab

Purpose:

- talk to the helper assistant or the internal chat model

What you can do here:

- ask what the next step is
- ask for draft SFT examples
- ask for validation examples
- ask the assistant to summarize job logs
- ask the assistant to help create corpus files

Behind the scenes, the helper runtime now receives a small cockpit protocol before each reply. That protocol tells it that it is inside llm-tweaker, that its job is to help with local model building and tuning, and that it should answer directly instead of dumping reasoning text.

If you see a protocol warning in the dashboard, that means the model replied with reasoning-only output anyway. That is a behavior mismatch from the model, not a broken button.

Good first prompts:

- `Review the current builder state and tell me the next step.`
- `Draft 12 chat training examples into chat_train.jsonl that teach explicit uncertainty.`
- `Create chat_val.jsonl with 8 hold-out examples that test handoffs and missing constraints.`

### assistant_sandbox Files

Purpose:

- working area for notes and draft datasets

Typical files:

- `chat_train.jsonl`
- `chat_val.jsonl`
- notes
- scratch prompts

### Sandbox Editor

Purpose:

- create and edit sandbox files directly

Use this for:

- SFT data
- notes
- prompt drafts

### local_corpus Files

Purpose:

- view the corpus files used for tokenizer and base-model training

Supported types include:

- text files
- JSON
- JSONL
- code files
- parquet

### Corpus Editor

Purpose:

- create or inspect local corpus files

How it works:

- text, JSON, and JSONL can be edited directly
- parquet files can be previewed
- new parquet files can be created from structured JSON object records

Beginner tip:

- if you do not understand parquet yet, start with `.txt` or `.jsonl`
- once the pipeline works, move structured datasets into `.parquet` if useful

Useful search terms:

- `what is parquet file`
- `jsonl vs parquet`
- `pyarrow parquet tutorial`

### Process Log

Purpose:

- see what actually happened

Use this when:

- a job failed
- the runtime did not start
- a file did not seem to save
- you are unsure what the last action did

If you are stuck, always read the Process Log before changing ten settings at once.

## Step 7: The Simplest End-to-End First Run

If you want the safest first loop, do this:

1. Install Python.
2. Install the required packages.
3. Put one real `.gguf` into `assistant_models/`.
4. Put a small real corpus into `local_corpus/`.
5. Launch the builder.
6. Choose `Truth-First Teammate`.
7. Save the design.
8. Publish the identity file.
9. Start the local runtime assistant.
10. Glance at the ECG window so you know what “idle but healthy” looks like before you launch work.
11. In Conversation Lab, ask it to draft `chat_train.jsonl`.
12. Ask it to draft `chat_val.jsonl`.
13. Review both files.
14. Train the tokenizer.
15. Run a small base training job.
16. While it runs, use the ECG window plus Process Log to confirm the box is still alive.
17. Run chat SFT.
18. Run the benchmark.
19. Test the result in Conversation Lab.

That is the complete first loop.

## Step 8: What Files You Need For Each Stage

### To Start The Dashboard

Needed:

- Python
- `fastapi`
- `uvicorn`

### To Start The Helper Assistant

Needed:

- a real `.gguf`
- runtime files under `runtime/windows/`

### To Train The Tokenizer

Needed:

- local corpus files

Optional:

- `pyarrow` for `.parquet`
- optional but recommended on Windows CPU training paths: Visual Studio Build Tools with Desktop C++

### To Train The Base Model

Needed:

- tokenizer
- local corpus
- PyTorch

### To Run Chat SFT

Needed:

- base checkpoint
- active identity file
- `chat_train.jsonl`
- `chat_val.jsonl`

### To Use Parquet In The Corpus

Needed:

- `pyarrow`

Install command:

```powershell
python -m pip install pyarrow
```

## Step 9: Common Beginner Failures

### Problem: The launcher says packages are missing

Fix:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install fastapi uvicorn
```

### Problem: No GGUF model appears

Check:

- the file is really in `assistant_models/`
- it is not zero bytes
- the filename ends in `.gguf`

### Problem: Corpus count is zero

Check:

- files are really inside `local_corpus/`
- they are supported formats
- if using split folders, use `local_corpus/train/`

### Problem: Parquet files are not working

Fix:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install pyarrow
```

If training later fails on Windows compile/build steps instead of Python imports, search for:

- `Visual Studio Build Tools download`
- `Desktop development with C++ workload`
- `torch compile windows cl.exe not found`

### Problem: Training runs out of memory

Reduce:

- `Device Batch Size`
- `Max Sequence Length`
- `Depth`

### Problem: The assistant drafts weak examples

Use more specific prompts.

Weak:

- `Make this better.`

Better:

- `Draft 10 short examples that teach the model to admit uncertainty, ask for missing constraints, and behave like one teammate among many.`

## Step 10: Search Terms Cheat Sheet

If you get stuck, these search terms are a good start:

- `python add to path windows`
- `powershell activate venv execution policy`
- `pip install pyarrow windows`
- `pytorch windows cpu install`
- `pytorch windows cuda install`
- `what is gguf`
- `small instruct gguf model`
- `what is parquet`
- `jsonl chat fine tuning format`
- `llama.cpp windows vulkan`
- `python fastapi uvicorn local web app`

## Final Advice

Do not try to make the model amazing on day one.

Your first success condition is much smaller:

- the dashboard starts
- the helper runtime starts
- the corpus is detected
- the tokenizer trains
- the base run completes
- SFT runs
- you can test the result

If you can do that once, you have the full loop working.

After that, improvement becomes much easier.
