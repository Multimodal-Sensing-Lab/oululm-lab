"""Minimal character-level GPT-style language model for teaching.

This is intentionally small and readable. It is suitable for explaining the
training loop, attention, loss, and sampling. It is not intended to be a strong
LLM implementation.
"""

from __future__ import annotations

import argparse
import math
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from .io import assert_disjoint, fresh_directory, load_config, read_jsonl, sha256, write_json


@dataclass
class TinyGPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TinyGPTConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(1, 1, cfg.block_size, cfg.block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, channels = x.shape
        # Each position produces a query (what to look for), key (how it can be
        # matched), and value (the information to mix). These are learned projections.
        q, k, v = self.qkv(x).split(channels, dim=2)
        q = q.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        # Compare each query with all keys in its head, then block future positions.
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :time, :time] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v  # Weighted mixture of allowed value vectors, not vocabulary probabilities.
        y = y.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.dropout(self.proj(y))


class Block(nn.Module):
    def __init__(self, cfg: TinyGPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # A residual connection keeps the old representation and adds a learned update.
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: TinyGPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        _, time = idx.shape
        if time == 0 or time > self.cfg.block_size:
            raise ValueError("Sequence length exceeds block_size")
        pos = torch.arange(0, time, device=idx.device)
        # [batch, positions] IDs become [batch, positions, features] vectors.
        x = self.token_embedding(idx) + self.position_embedding(pos)
        x = self.blocks(x)
        # One score per vocabulary token at EVERY input position.
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            # Mean -log(probability of the observed next token), across batch/positions.
            # Computing loss alone does not train: the caller must backpropagate/update.
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8) -> torch.Tensor:
        if temperature < 0 or idx.shape[1] == 0 or max_new_tokens < 0:
            raise ValueError("Need a nonempty prompt, nonnegative temperature and token budget")
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            idx_next = (logits.argmax(dim=-1, keepdim=True) if temperature == 0 else
                        torch.multinomial(F.softmax(logits / temperature, dim=-1), num_samples=1))
            idx = torch.cat((idx, idx_next), dim=1)
        self.train(was_training)
        return idx


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str):
    if len(data) <= block_size or min(batch_size, block_size) < 1:
        raise ValueError("Each split needs more tokens than block_size; batch/block sizes must be positive")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def encode(text, stoi, strict=False):
    unknown = set(text) - set(stoi)
    if unknown and strict:
        raise ValueError(f"Characters absent from training vocabulary: {sorted(unknown)!r}")
    return [stoi.get(char, 0) for char in text]


def load_checkpoint(path, device="cpu"):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") == "llm-workshop-word-v1":
        raise ValueError("This is a word checkpoint. Use python -m llm_workshop.words.demo instead of the character tools.")
    model = TinyGPT(TinyGPTConfig(**state["config"])).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, state


def train(cfg, resume=None):
    torch.set_num_threads(int(cfg.get("num_threads", 2)))
    torch.manual_seed(int(cfg.get("seed", 1337)))
    device = cfg.get("device", "cpu")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; choose device: cpu explicitly")
    for key in ("batch_size", "block_size", "max_steps", "eval_interval"):
        if int(cfg[key]) < 1:
            raise ValueError(f"{key} must be positive")
    if "train_file" in cfg:
        split_rows = {name: read_jsonl(cfg[key]) for name, key in
                      (("train", "train_file"), ("validation", "validation_file"))}
        assert_disjoint(split_rows)
        texts = {name: "\n\n".join(row["text"] for row in rows) for name, rows in split_rows.items()}
        hashes = {key: sha256(cfg[key]) for key in ("train_file", "validation_file")}
    else:
        documents = Path(cfg["data_path"]).read_text(encoding="utf-8").split("\n\n")
        documents = list(dict.fromkeys(doc.strip() for doc in documents if doc.strip()))
        if len(documents) < 2:
            raise ValueError("Supply distinct documents separated by blank lines or explicit JSONL splits")
        boundary = max(1, int(len(documents) * .9))
        texts = {"train": "\n\n".join(documents[:boundary]), "validation": "\n\n".join(documents[boundary:])}
        hashes = {"data_path": sha256(cfg["data_path"])}
    chars = ["<UNK>"] + sorted(set(texts["train"]))
    stoi = {char: i for i, char in enumerate(chars)}
    model_cfg = TinyGPTConfig(len(chars), **{k: cfg[k] for k in
                               ("block_size", "n_layer", "n_head", "n_embd", "dropout")})
    model = TinyGPT(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
    start_step = 0
    output = Path(cfg["output_dir"])
    if resume:
        model, state = load_checkpoint(resume, device)
        if state["config"] != model_cfg.__dict__ or state["data_hashes"] != hashes:
            raise ValueError("Resume model/data differ from checkpoint")
        for key in ("batch_size", "learning_rate", "seed"):
            if cfg.get(key) != state["run_config"].get(key):
                raise ValueError(f"Resume changes {key}; start a new experiment instead")
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
        optimizer.load_state_dict(state["optimizer_state_dict"])
        torch.set_rng_state(state["rng_state"])
        if str(device).startswith("cuda") and state["cuda_rng_state"]:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        start_step = state["step"]
        output.mkdir(parents=True, exist_ok=True)
    else:
        fresh_directory(output)
    if start_step >= int(cfg["max_steps"]):
        raise ValueError("max_steps must exceed the resumed step")
    tensors = {name: torch.tensor(encode(text, stoi), dtype=torch.long) for name, text in texts.items()}
    block = int(cfg["block_size"])
    if any(len(values) <= block for values in tensors.values()):
        raise ValueError("Each split must contain more characters than block_size")
    if int(cfg.get("eval_batches", 4)) < 1:
        raise ValueError("eval_batches must be positive")
    # Fixed validation windows do not consume training RNG.
    starts = torch.linspace(0, len(tensors["validation"]) - block - 1,
                            steps=int(cfg.get("eval_batches", 4))).long().tolist()
    def evaluate():
        model.eval()
        with torch.no_grad():
            losses = [model(tensors["validation"][i:i+block].unsqueeze(0).to(device),
                            tensors["validation"][i+1:i+block+1].unsqueeze(0).to(device))[1].item()
                      for i in starts]
        model.train()
        return sum(losses) / len(losses)

    prompt = cfg.get("generate_prompt", "Once upon a time")
    prompt_ids = encode(prompt, stoi, strict=True)
    if not prompt_ids:
        raise ValueError("Empty generation prompt")
    write_json(output / "run_config.json", cfg)
    print(f"device={device}; parameters={sum(p.numel() for p in model.parameters())}; validation_loss={evaluate():.4f}")
    with (output / "metrics.jsonl").open("a" if resume else "w", encoding="utf-8") as log:
        for step in range(start_step + 1, int(cfg["max_steps"]) + 1):
            xb, yb = get_batch(tensors["train"], int(cfg["batch_size"]), block, device)
            _, loss = model(xb, yb)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step % int(cfg["eval_interval"]) == 0 or step == int(cfg["max_steps"]):
                metrics = {"step": step, "train_loss": loss.item(), "validation_loss": evaluate()}
                print(metrics)
                log.write(json.dumps(metrics) + "\n")
                log.flush()
                state = {"model_state_dict": model.state_dict(), "config": model_cfg.__dict__,
                         "stoi": stoi, "itos": dict(enumerate(chars)), "step": step,
                         "optimizer_state_dict": optimizer.state_dict(), "rng_state": torch.get_rng_state(),
                         "cuda_rng_state": torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else [],
                         "run_config": cfg, "data_hashes": hashes}
                temporary = output / "checkpoint.tmp"
                torch.save(state, temporary)
                temporary.replace(output / "checkpoint.pt")
    ids = torch.tensor([prompt_ids], device=device)
    result = model.generate(ids, int(cfg.get("max_new_tokens", 100)), temperature=.8)[0].tolist()
    sample = "".join(chars[i] for i in result)
    (output / "sample.txt").write_text(sample, encoding="utf-8")
    print(sample)
    return model


CONFIG_KEYS = "train_file validation_file data_path output_dir device seed batch_size block_size max_steps eval_interval learning_rate n_layer n_head n_embd dropout generate_prompt max_new_tokens num_threads eval_batches".split()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/from_scratch_desktop.yml"))
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    train(load_config(args.config, CONFIG_KEYS), args.resume)


if __name__ == "__main__":
    main()
