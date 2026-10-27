"""Build a listable local symlink view without listing the model source.

The checkpoint directory may deny readdir while known file paths remain
readable.  Read config and the shard index directly, probe every required shard,
then link those known files into a local directory for Transformers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import time


ROOT = Path(__file__).resolve().parent
OPTIONAL_FILES = (
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "tokenizer.model",
)


def _safe_relative(name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Invalid checkpoint shard path: {name!r}")
    return Path(*relative.parts)


def _probe(path: Path) -> int:
    # Opening the known path is the authority: do not assume directory read
    # permission from exists(), access(), or a glob of the inaccessible source.
    with path.open("rb") as stream:
        if not stream.read(1):
            raise ValueError(f"Empty checkpoint/tokenizer file: {path}")
    return path.stat().st_size


def prepare(source: Path, destination: Path) -> dict:
    source = source.absolute()
    destination = destination.absolute()
    if source == destination:
        raise ValueError("The model view must be separate from the source directory")
    with (source / "config.json").open() as stream:
        config = json.load(stream)
    with (source / "model.safetensors.index.json").open() as stream:
        index = json.load(stream)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model.safetensors.index.json has no non-empty weight_map")
    shards = sorted({_safe_relative(name).as_posix() for name in weight_map.values()})
    required = ["config.json", "model.safetensors.index.json", *shards]
    probed = {name: _probe(source / name) for name in required}
    skipped = {}
    for name in OPTIONAL_FILES:
        try:
            probed[name] = _probe(source / name)
        except (FileNotFoundError, PermissionError) as exc:
            skipped[name] = type(exc).__name__

    # Check all collisions before adding links, and never replace an existing
    # file or a link pointing at some other model.
    for name in probed:
        target = destination / name
        expected = source / name
        if target.is_symlink():
            if target.readlink() != expected:
                raise FileExistsError(f"Conflicting model symlink: {target}")
        elif target.exists():
            raise FileExistsError(f"Refusing to replace an existing file: {target}")
    for name in probed:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_symlink():
            target.symlink_to(source / name)

    report = {
        "source": str(source),
        "destination": str(destination),
        "created_epoch": time.time(),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "shard_count": len(shards),
        "indexed_tensor_count": len(weight_map),
        "weight_bytes": sum(probed[name] for name in shards),
        "linked_file_sizes": probed,
        "skipped_optional_files": skipped,
        "source_directory_enumerated": False,
    }
    manifest = destination / "view_manifest.json"
    temporary = destination / ".view_manifest.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(manifest)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/mnt/nfs0/zhangh/models/Qwen3-32B"))
    parser.add_argument("--destination", type=Path, default=ROOT / "model-view" / "Qwen3-32B")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.destination), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
