"""JSONL, configuration, and provenance helpers (no model dependencies)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number}: expected an object")
                rows.append(row)
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    return rows


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(text):
    return " ".join(str(text).casefold().split())


def text_id(text):
    return hashlib.sha256(normalized(text).encode()).hexdigest()


def load_config(path, allowed):
    import yaml
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a YAML mapping")
    unknown = set(cfg) - set(allowed)
    if unknown:
        raise ValueError(f"Unrecognized configuration keys: {sorted(unknown)}")
    return cfg


def messages_for(row):
    if "messages" in row:
        messages = row["messages"]
    else:
        messages = row.get("prompt", []) + row.get("completion", [])
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("Expected a conversation ending with an assistant answer")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError("Invalid conversation role")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError("Empty/non-text message")
    if messages[-1]["role"] != "assistant" or messages[-2]["role"] != "user":
        raise ValueError("The final exchange must be user then assistant")
    return messages


def assert_disjoint(splits):
    """Reject cross-split IDs, groups, or identical inputs; allow within-split variants."""
    seen = {}
    for split, rows in splits.items():
        for row in rows:
            keys = [(kind, str(row[kind])) for kind in ("id", "group_id") if kind in row]
            if "text" in row:
                keys.append(("content", text_id(row["text"])))
            else:
                messages = messages_for(row)
                keys.append(("content", text_id(json.dumps(messages[:-1], sort_keys=True))))
            for key in keys:
                if key in seen and seen[key] != split:
                    raise ValueError(f"Cross-split overlap: {key} in {seen[key]} and {split}")
                seen[key] = split


def fresh_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output directory is not empty: {path}; choose a new run directory")
    path.mkdir(parents=True, exist_ok=True)
    return path
