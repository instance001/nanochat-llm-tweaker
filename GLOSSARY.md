# Glossary (Repo Excerpt)

For the full glossary, see: https://github.com/instance001/Whatisthisgithub/blob/main/GLOSSARY.md

This file contains only the glossary entries for this repository. Mapping tag legends and global notes live in the full glossary.

## nanochat-llm-tweaker
| Term | Alternate term(s) | Alt map | External map | Relation to existing terminology | What it is | What it is not | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LLM Tweaker Builder | nanochat-llm-tweaker, llm tweaker builder | ~ | ~ | Local builder dashboard wrapper / broader local LLM workflow suite | Local-first LLM build suite based on Andrej Karpathy's `nanochat`, adapted into a guided dashboard for corpus prep, tokenizer training, base-model training, chat SFT, checkpoint testing, and related local workflows, with a GGUF helper assistant and drop-in chatty-cog module packaging | Not a cloud training service; not just a narrow parameter-tweaking utility; not just the upstream nanochat repo with no wrapper changes | nanochat-llm-tweaker/README.md |
| Chatty-cog wrapper layer | wrapper layer, outer module wrapper | ~ | ~ | Host integration layer | Outer module folder that adds discovery metadata, hosted webview loading, bridge-side workspace fields, and suspend-rundown integration for chatty-cog hosting | Not the training dashboard logic itself; not the upstream nanochat codebase | nanochat-llm-tweaker/README.md |
| `nanochat-master/` | nested builder app, upstream-derived app | = | ~ | Nested upstream-derived application | Directory containing the actual local builder application and upstream-derived codebase that the wrapper hosts | Not just static assets; not limited to chatty-cog metadata | nanochat-llm-tweaker/README.md |
| GGUF helper assistant | helper `.gguf` | ~ | ~ | Local helper-model slot | Local assistant model placed in `nanochat-master/Assistant_models/` to support the guided builder flow | Not the base model being trained; not a remote API helper | nanochat-llm-tweaker/README.md |
| Local corpus path | real corpus, `local_corpus/` | = | ~ | Corpus intake location | User-provided corpus folder under `nanochat-master/local_corpus/` used for local corpus prep and downstream builder steps | Not a bundled training corpus; not a cloud dataset source | nanochat-llm-tweaker/README.md |
| Licensing boundary | outer wrapper vs nested app licensing | ~ | ~ | Mixed-license packaging boundary | Explicit split where the outer chatty-cog integration layer is intended for AGPLv3 distribution while the nested upstream-derived `nanochat-master/` app carries its own MIT license | Not a single undifferentiated repo license; not permission to drop upstream attribution | nanochat-llm-tweaker/README.md; nanochat-llm-tweaker/LICENSES.md |
