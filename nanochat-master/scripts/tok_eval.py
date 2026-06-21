"""
Evaluate local tokenizer compression on built-in samples and the local corpus.
"""

from __future__ import annotations

import argparse

from nanochat.dataset import get_local_corpus_dir, parquets_iter_batched
from nanochat.tokenizer import get_tokenizer

parser = argparse.ArgumentParser(description="Evaluate tokenizer compression locally")
parser.add_argument("--corpus-dir", type=str, default="", help="Local corpus directory (default: local_corpus or NANOCHAT_LOCAL_CORPUS_DIR)")
parser.add_argument("--max-corpus-docs", type=int, default=8, help="Number of train/val documents to sample from the local corpus")
args = parser.parse_args()

tokenizer = get_tokenizer()

samples = [
    (
        "builder-note",
        "The assistant should prefer correctness over speed, admit uncertainty, and behave like one teammate among many.",
    ),
    (
        "code",
        "def route_request(path, method):\n    if method == 'POST' and path.startswith('/api/'):\n        return 'ok'\n",
    ),
    (
        "math",
        "If the answer cannot be verified from the available evidence, say that clearly and outline the next check.",
    ),
]

for split in ("train", "val"):
    try:
        docs = []
        for batch in parquets_iter_batched(split=split, data_dir=args.corpus_dir or None, batch_size=max(1, args.max_corpus_docs)):
            docs.extend(batch)
            if len(docs) >= args.max_corpus_docs:
                break
        docs = docs[: args.max_corpus_docs]
        for index, doc in enumerate(docs):
            samples.append((f"{split}-{index}", doc))
    except AssertionError:
        pass

print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")
print(f"Corpus dir: {get_local_corpus_dir(args.corpus_dir or None)}")
print()
print(f"{'Sample':<18} {'Bytes':>8} {'Tokens':>8} {'Bytes/Token':>12}")
print("-" * 52)

report_rows = []
for name, text in samples:
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text
    byte_count = len(text.encode("utf-8"))
    token_count = len(encoded)
    ratio = byte_count / max(1, token_count)
    print(f"{name:<18} {byte_count:>8} {token_count:>8} {ratio:>12.2f}")
    report_rows.append(
        {
            "sample": name,
            "bytes": byte_count,
            "tokens": token_count,
            "bytes_per_token": ratio,
        }
    )

from nanochat.report import get_report

get_report().log(section="Tokenizer evaluation", data=report_rows)
