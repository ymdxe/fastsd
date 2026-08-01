#!/usr/bin/env python3
import argparse
import sys
from typing import Dict, List, Tuple

from transformers import AutoTokenizer


def resolve_model(name_or_path: str, use_model_zoo: bool) -> str:
    if not use_model_zoo:
        return name_or_path

    # Reuse project model mapping logic.
    from src.util import model_zoo

    class _Args:
        def __init__(self, draft_model: str, target_model: str):
            self.draft_model = draft_model
            self.target_model = target_model
            self.vocab_size = None

    args = _Args(name_or_path, name_or_path)
    model_zoo(args)
    return args.draft_model


def sample_diffs(
    vocab_a: Dict[str, int], vocab_b: Dict[str, int], max_show: int
) -> Tuple[List[str], List[str], List[Tuple[str, int, int]]]:
    only_a = []
    only_b = []
    id_mismatch = []

    for token, idx in vocab_a.items():
        if token not in vocab_b:
            if len(only_a) < max_show:
                only_a.append(token)
            continue
        if vocab_b[token] != idx and len(id_mismatch) < max_show:
            id_mismatch.append((token, idx, vocab_b[token]))

    for token in vocab_b:
        if token not in vocab_a and len(only_b) < max_show:
            only_b.append(token)

    return only_a, only_b, id_mismatch


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether draft/cloud(target) tokenizers are consistent."
    )
    parser.add_argument("--draft_model", required=True, help="Draft model name/path")
    parser.add_argument("--target_model", required=True, help="Target model name/path")
    parser.add_argument(
        "--use_model_zoo",
        action="store_true",
        help="Resolve --draft_model/--target_model via src.util.model_zoo",
    )
    parser.add_argument(
        "--max_show",
        type=int,
        default=10,
        help="Maximum number of sample differences to print per category",
    )
    args = parser.parse_args()

    draft_model = resolve_model(args.draft_model, args.use_model_zoo)
    target_model = resolve_model(args.target_model, args.use_model_zoo)

    print(f"[INFO] loading draft tokenizer:  {draft_model}")
    tok_draft = AutoTokenizer.from_pretrained(draft_model, trust_remote_code=True)
    print(f"[INFO] loading target tokenizer: {target_model}")
    tok_target = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True)

    vocab_draft = tok_draft.get_vocab()
    vocab_target = tok_target.get_vocab()

    same_vocab = vocab_draft == vocab_target
    same_vocab_size = len(vocab_draft) == len(vocab_target)
    same_special_map = tok_draft.special_tokens_map == tok_target.special_tokens_map

    print("\n=== Tokenizer Consistency Report ===")
    print(f"draft tokenizer class:  {tok_draft.__class__.__name__}")
    print(f"target tokenizer class: {tok_target.__class__.__name__}")
    print(f"draft vocab size:       {len(vocab_draft)}")
    print(f"target vocab size:      {len(vocab_target)}")
    print(f"same vocab size:        {same_vocab_size}")
    print(f"same special tokens:    {same_special_map}")
    print(f"full vocab identical:   {same_vocab}")

    test_ids = [0, 1, 2, 7, 10, 13, 16, 17, 18, 19, 20, 21, 22, 185, 315]
    print("\n[id decode samples]")
    for tid in test_ids:
        d_txt = repr(tok_draft.decode([tid], skip_special_tokens=False))
        t_txt = repr(tok_target.decode([tid], skip_special_tokens=False))
        mark = "OK" if d_txt == t_txt else "DIFF"
        print(f"id={tid:>5} draft={d_txt:<16} target={t_txt:<16} {mark}")

    test_text = 'def fib(n):\n    """Returns the nth Fibonacci number."""\n    '
    d_ids = tok_draft.encode(test_text, add_special_tokens=False)
    t_ids = tok_target.encode(test_text, add_special_tokens=False)
    print("\n[text encode sample]")
    print(f"sample text: {repr(test_text)}")
    print(f"draft ids len={len(d_ids)}")
    print(f"target ids len={len(t_ids)}")
    print(f"encode identical: {d_ids == t_ids}")
    if d_ids != t_ids:
        print(f"draft ids (head 32):  {d_ids[:32]}")
        print(f"target ids (head 32): {t_ids[:32]}")

    if not same_vocab:
        only_d, only_t, mismatch = sample_diffs(vocab_draft, vocab_target, args.max_show)
        print("\n[vocab differences]")
        print(f"tokens only in draft (sample):  {only_d}")
        print(f"tokens only in target (sample): {only_t}")
        print(f"same token but id mismatch (sample): {mismatch}")

    # Return non-zero to make CI/script checks easy.
    return 0 if (same_vocab and same_special_map and d_ids == t_ids) else 1


if __name__ == "__main__":
    sys.exit(main())

