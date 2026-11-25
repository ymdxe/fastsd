"""Load official model implementations by architecture, not checkpoint name."""

import hashlib
import json


def model_config(path):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(path)
    if config.model_type not in {"qwen3", "llama"}:
        raise ValueError(f"SpecEdge architecture {config.model_type!r} unsupported for {path}")
    return config


def model_class(config):
    from model.llama import LlamaForCausalLM
    from model.qwen3 import Qwen3ForCausalLM

    return {"qwen3": Qwen3ForCausalLM, "llama": LlamaForCausalLM}[config.model_type]


def load_model(path, config, device, dtype):
    return model_class(config).from_pretrained(path, config=config, torch_dtype=dtype, device_map=str(device)).eval()


def token_signature(tokenizer, config, generation=None):
    vocab = tokenizer.get_vocab()
    if not vocab or min(vocab.values()) < 0 or max(vocab.values()) >= config.vocab_size:
        raise ValueError("tokenizer vocabulary IDs exceed model vocabulary")

    def ids(value):
        return sorted(set(value)) if isinstance(value, (tuple, list)) else ([] if value is None else [value])

    result = {"vocab_size": config.vocab_size, "vocab_sha256": hashlib.sha256(
        json.dumps(vocab, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
    ).hexdigest()}
    for name in ("eos", "pad", "bos"):
        attr = f"{name}_token_id"
        token_ids = ids(getattr(tokenizer, attr, None))
        config_ids = ids(getattr(config, attr, None))
        generation_ids = ids(getattr(generation, attr, None))
        effective = generation_ids or config_ids or token_ids
        if not set(token_ids + config_ids).issubset(effective) or any(
            not 0 <= token_id < config.vocab_size for token_id in effective
        ):
            raise ValueError(f"incompatible {attr}: tokenizer={token_ids}, model={config_ids}, generation={generation_ids}")
        if name == "eos" and not effective:
            raise ValueError("model/tokenizer must define EOS")
        result[name] = {"effective": effective, "tokenizer": token_ids}
    return result


def compare_token_signatures(target, draft):
    differences = [key for key in target if target[key] != draft.get(key)]
    if differences:
        raise ValueError(f"target/draft token mapping is incompatible: {', '.join(differences)}")


class PromptTokenizer:
    def __init__(self, tokenizer, add_special_tokens):
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens

    def encode(self, text, **kwargs):
        kwargs.setdefault("add_special_tokens", self.add_special_tokens)
        return self.tokenizer.encode(text, **kwargs)

    def __getattr__(self, attr):
        return getattr(self.tokenizer, attr)


def load_tokenizer(model_path, tokenizer_path, config, *, chat=False):
    from transformers import AutoTokenizer, GenerationConfig

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, legacy=False)
    try:
        generation = GenerationConfig.from_pretrained(model_path)
    except OSError:
        generation = None
    signature = token_signature(tokenizer, config, generation)
    if chat and not tokenizer.chat_template:
        raise ValueError(f"chat template required for {tokenizer_path}")
    return PromptTokenizer(tokenizer, add_special_tokens=not chat), signature
