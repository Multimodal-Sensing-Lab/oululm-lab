"""Prepare an offline fact experiment or small public datasets with frozen splits."""
from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

from .io import assert_disjoint, fresh_directory, sha256, text_id, write_json, write_jsonl


def record(identifier, group, question, answer, task, **metadata):
    return {"id": identifier, "group_id": group, "task": task,
            "messages": [{"role": "user", "content": question},
                         {"role": "assistant", "content": answer}], **metadata}


def finish(output, splits, provenance):
    assert_disjoint(splits)
    files = {}
    for name, rows in splits.items():
        if not rows:
            raise ValueError(f"Empty {name} split")
        path = output / f"{name}.jsonl"
        write_jsonl(path, rows)
        files[name] = {"records": len(rows), "sha256": sha256(path),
                       "labels": dict(Counter(r["messages"][-1]["content"] for r in rows))
                       if rows[0].get("task") in {"finance", "biology"} else None}
    write_json(output / "manifest.json", {**provenance, "files": files})


def handbook(output, seed=42, assignment="a"):
    """Same entities/splits in A and B, independently reassigned room values."""
    output = fresh_directory(output)
    rng = random.Random(seed)
    names = [f"{a}{b}" for a in ("Ner", "Tal", "Vor", "Lum", "Zev", "Pav", "Kyr", "Bel", "Dax", "Sov")
             for b in ("al", "in", "or", "um", "ek", "is", "av", "on", "ul", "et")]
    rng.shuffle(names)
    rooms = [f"K{i}" for i in range(101, 201)]
    random.Random(seed + (0 if assignment == "a" else 1009)).shuffle(rooms)
    templates = ["Where is the {name} instrument stored?", "Give the storage room for {name}.",
                 "In which room can I find the instrument {name}?", "What is the room code for {name}?"]
    prefix = "In the fictional Aurora Teaching Lab, answer with only the room code. If unknown, answer UNKNOWN.\n"
    splits = {"train": [], "validation": [], "test": []}
    docs_v1, docs_v2, facts, recall, updates, prose = [], [], [], [], [], []
    for i, (name, room) in enumerate(zip(names, rooms)):
        group = f"instrument-{i:03d}"
        split = "train" if i < 60 else "validation" if i < 80 else "test"
        new_room = f"M{int(room[1:]) + 200}"
        facts.append({"id": group, "name": name, "room_v1": room, "room_v2": new_room, "split": split})
        docs_v1.append({"id": group, "text": f"Aurora Teaching Lab (fictional): The {name} instrument is stored in room {room}."})
        docs_v2.append({"id": group, "text": f"Aurora Teaching Lab (fictional), updated handbook: The {name} instrument is stored in room {new_room}."})
        questions = templates if split == "train" else templates[:1]
        for j, template in enumerate(questions):
            splits[split].append(record(f"{group}-{j}", group, prefix + template.format(name=name), room,
                                        "knowledge", source_id=group,
                                        test_group="memorization" if split == "train" else "unseen_facts"))
        if split == "train":
            prose.append({"id": group, "group_id": group, "text": docs_v1[-1]["text"]})
            question = prefix + f"Which room should I visit to collect {name}?"
            recall.append(record(f"{group}-paraphrase", group, question, room, "knowledge",
                                 source_id=group, test_group="seen_fact_new_wording"))
            updates.append(record(f"{group}-update", group, question, new_room, "knowledge",
                                  source_id=group, test_group="updated_seen_facts"))
    finish(output, splits, {"source": "workshop-generated fictional facts; no real policy",
                            "seed": seed, "assignment": assignment,
                            "split_method": "60/20/20 disjoint entities; paraphrase/update diagnostics are separate"})
    for filename, rows in {"facts": facts, "documents_v1": docs_v1, "documents_v2": docs_v2,
                           "test_paraphrase": recall, "test_updates": updates,
                           "train_prose": prose,
                           "validation_prose": [{"id": d["id"], "group_id": d["id"], "text": d["text"]}
                                                for d in docs_v1[60:80]]}.items():
        write_jsonl(output / f"{filename}.jsonl", rows)
    print(f"Prepared fictional handbook {assignment}: {output}")


def public_dataset(output, dataset, limit=None, seed=42, revision=None):
    # Imports stay local so handbook preparation works without Hugging Face packages/network.
    from datasets import load_dataset
    from huggingface_hub import HfApi
    output = fresh_directory(output)
    source = {"finance": "lmassaron/FinancialPhraseBank", "biology": "qiaojin/PubMedQA",
              "tinystories": "roneneldan/TinyStories"}[dataset]
    revision = HfApi().dataset_info(source, revision=revision).sha
    license_name = {"finance": "CC-BY-NC-SA-4.0", "biology": "MIT (dataset card)",
                    "tinystories": "CDLA-Sharing-1.0"}[dataset]
    rng = random.Random(seed)
    splits = {"train": [], "validation": [], "test": []}
    duplicates = 0
    seen = set()

    def take(split, subset=None):
        stream = load_dataset(source, subset, split=split, revision=revision, streaming=True)
        if limit:
            # Bounded shuffle: practical sample, explicitly not a global uniform sample.
            stream = stream.shuffle(seed=seed, buffer_size=max(1000, limit))
            return list(stream.take(limit))
        return list(stream)

    if dataset == "finance":
        # Keep source split assignment. Remove duplicated inputs from later splits.
        for split in splits:
            for row in take(split):
                sentence, label = row["sentence"].strip(), str(row["sentiment"]).lower().strip()
                if not sentence or label not in {"positive", "neutral", "negative"}:
                    raise ValueError("Unexpected FinancialPhraseBank row")
                group = text_id(sentence)
                if group in seen:
                    duplicates += 1
                    continue
                seen.add(group)
                splits[split].append(record(group, group,
                    "Classify the financial sentiment as positive, neutral, or negative. Return only the label.\n\nText: " + sentence,
                    label, dataset))
    elif dataset == "biology":
        groups = defaultdict(list)
        for row in take("train", "pqa_labeled"):
            label = row["final_decision"].lower().strip()
            if label not in {"yes", "no", "maybe"}:
                raise ValueError("Unexpected PubMedQA decision")
            group = str(row["pubid"])
            if group in seen:
                duplicates += 1
                continue
            seen.add(group)
            context = "\n".join(row["context"]["contexts"])
            question = row["question"].strip()
            if not context or not question:
                raise ValueError("Empty PubMedQA question/context")
            groups[label].append(record(group, group,
                "Using only this abstract, answer yes, no, or maybe. Return only the label.\n\nQuestion: "
                + question + "\n\nAbstract: " + context, label, dataset))
        for rows in groups.values():
            if len(rows) < 3:
                raise ValueError("Need at least 3 examples per label; increase --limit")
            rng.shuffle(rows)
            nval = ntest = max(1, int(len(rows) * .15))
            splits["validation"].extend(rows[:nval])
            splits["test"].extend(rows[nval:nval + ntest])
            splits["train"].extend(rows[nval + ntest:])
    else:
        rows = take("train")
        rng.shuffle(rows)
        boundary = max(1, int(len(rows) * .1))
        sources = {"validation": rows[:boundary], "train": rows[boundary:], "test": take("validation")}
        for split, source_rows in sources.items():
            for row in source_rows:
                text = row["text"].strip()
                if not text:
                    continue
                group = text_id(text)
                if group in seen:
                    duplicates += 1
                    continue
                seen.add(group)
                splits[split].append({"id": group, "group_id": group, "text": text})
    finish(output, splits, {"source": source, "revision": revision, "license": license_name,
                            "seed": seed, "limit_per_source_split": limit, "duplicates_removed": duplicates,
                            "sampling": "bounded streaming shuffle" if limit else "full source splits",
                            "split_method": "source splits" if dataset == "finance" else
                            "stratified by decision and grouped by PubMed ID" if dataset == "biology" else
                            "source training documents split 90/10; source validation reserved for test"})
    print(f"Prepared {dataset}: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["handbook", "finance", "biology", "tinystories"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="Positive per-source-split sample limit; use 1000 for TinyStories")
    parser.add_argument("--revision")
    parser.add_argument("--assignment", choices=["a", "b"], default="a")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.dataset == "tinystories" and args.limit is None:
        parser.error("Set --limit explicitly to avoid loading millions of stories into RAM")
    if args.dataset == "handbook":
        handbook(args.output_dir, args.seed, args.assignment)
    else:
        public_dataset(args.output_dir, args.dataset, args.limit, args.seed, args.revision)


if __name__ == "__main__":
    main()
