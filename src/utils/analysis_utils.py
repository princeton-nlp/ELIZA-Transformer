"""Utilities for analyzing ELIZA errors."""

from collections import deque
import json
from pathlib import Path

import pandas as pd
from src import simple_tokenizers, data_utils


def annotate_predictions(predictions):
    predictions["template_length"] = [len(t.split()) for t in predictions["template"]]
    predictions["num_wildcards"] = [t.count("0") for t in predictions["template"]]
    predictions["transformation_length"] = [
        len(r.split()) for r in predictions["transformation"]
    ]
    predictions["num_copies"] = [
        sum(g.isnumeric() for g in r.split()) for r in predictions["transformation"]
    ]
    predictions["input_length"] = [len(s.split()) for s in predictions["input"]]
    predictions["sentence"] = [s[: s.find(" E")] for s in predictions["input"]]
    return predictions


def annotate_copy_length(predictions):
    predictions["output_length"] = [len(t.split()) + 1 for t in predictions["want"]]
    predictions["copy_length"] = [
        o_len - (t_len - n_copies)
        for o_len, t_len, n_copies in zip(
            predictions["output_length"],
            predictions["transformation_length"],
            predictions["num_copies"],
        )
    ]
    return predictions


def get_predictions(output_dir, epoch=None):
    output_dir = Path(output_dir)
    with open(output_dir / "args.json", "r") as f:
        args = json.load(f)
    if epoch is None:
        path = sorted(list(output_dir.glob("predictions_*")))[-1]
    elif type(epoch) == str:
        path = output_dir / f"predictions_{epoch}.csv"
    else:
        path = output_dir / f"predictions_{epoch:02d}.csv"
    predictions = pd.read_csv(path)
    return annotate_predictions(predictions), args


def get_all_predictions(output_dir):
    output_dir = Path(output_dir)
    with open(output_dir / "args.json", "r") as f:
        args = json.load(f)
    lst = []
    for path in output_dir.glob("predictions_*"):
        lst.append(pd.read_csv(path))
    predictions = pd.concat(lst)
    return annotate_predictions(predictions), args


def get_output_dirs(base):
    return list(base.glob("*/*/*"))


def parse_data_dir(s):
    parts = s.split("/")[-1].split("_")
    d = {}
    for part in parts:
        idxs = [i for i, c in enumerate(part) if c.isnumeric()]
        if not idxs:
            continue
        i = idxs[0]
        k, v = part[:i], part[i:]
        v = float(v) if "." in v else int(v)
        d[k] = v
    return d


def get_data_args(data_dir):
    fn = Path(data_dir) / "args.json"
    with open(fn, "r") as f:
        args = json.load(f)
    return args


def add_data_args(data_dir, df):
    data_args = get_data_args(data_dir)
    for k, v in data_args.items():
        df[k] = v
    return df


def get_metrics(paths):
    rows = []
    for i, path in enumerate(paths):
        if not (path / "args.json").exists() or not (path / "metrics.csv").exists():
            continue
        with open(path / "args.json", "r") as f:
            args = json.load(f)
        metrics = pd.read_csv(path / "metrics.csv")
        for k, v in args.items():
            metrics[k] = v
        data_args = get_data_args(args["data_dir"])
        for k, v in data_args.items():
            metrics[k] = v
        rows.append(metrics)
    print(f"Got metrics for {len(rows)} paths")
    return pd.concat(rows)


def get_memory_stats(path):
    script = pd.read_csv(path / "script.csv")
    memory_template = script.query("type == 'memory'").iloc[0]["template"]
    memory_template_id = str(
        script.query(f"template == '{memory_template}'").iloc[0]["template_id"]
    )
    null_id = str(script.query(f"type == 'none'").iloc[0]["template_id"])
    tokenizer, idx_w = simple_tokenizers.get_tokenizer(path)
    val_df = data_utils.load_dataset(path / "validation.csv", tokenizer)
    rows = []
    for conv_id, templates, conv in zip(
        val_df["conv_id"], val_df["template_ids"], val_df["input"]
    ):
        turns = conv.split(". U")
        turns = [turns[0]] + [". U" + s for s in turns[1:]]
        queue = deque()
        num_enqueues = 0
        num_dequeues = 0
        position = 0
        for i, (template_id, turn) in enumerate(zip(templates.split(";"), turns)):
            row = {
                "conv_id": conv_id,
                "turn": i,
                "position": position,
                "enqueues": num_enqueues,
                "dequeues": num_dequeues,
            }
            row.update(
                {
                    k: 0
                    for k in (
                        "tgt_turn",
                        "tgt_pos",
                        "queue_size",
                        "num_enqueues",
                        "num_dequeues",
                    )
                }
            )
            if template_id == memory_template_id:
                queue.append((i, position))
                num_enqueues += 1
                row["kind"] = "enqueue"
            elif template_id == null_id and len(queue):
                tgt_turn, tgt_pos = queue.popleft()
                row.update(
                    {
                        "kind": "dequeue",
                        "tgt_turn": tgt_turn,
                        "tgt_pos": tgt_pos,
                        "queue_size": len(queue) + 1,
                        "num_enqueues": num_enqueues,
                        "num_dequeues": num_dequeues,
                    }
                )
                num_dequeues += 1
            elif template_id == null_id:
                row.update(
                    {
                        "kind": "none",
                        "num_enqueues": num_enqueues,
                        "num_dequeues": num_dequeues,
                    }
                )
            else:
                row["kind"] = "-"
            rows.append(row)
            position += turn.find(" ") + 1
    return pd.DataFrame(rows)


def add_memory_stats(preds, memory_df):
    for col in [
        "position",
        "enqueues",
        "dequeues",
        "tgt_turn",
        "tgt_pos",
        "queue_size",
        "num_enqueues",
        "num_dequeues",
        "kind",
    ]:
        preds[col] = memory_df[col]
    return preds
