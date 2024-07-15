import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import GPT2Config
from tqdm import tqdm

from src import data_utils, generate_data, models, simple_tokenizers
from src.utils import logging

logger = logging.get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="output/scratch")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--attention_layer", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_rows", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = Path(args.model_dir) / "probing"
        if not args.output_dir.exists():
            args.output_dir.mkdir(parents=True)
    return args


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_model_and_tokenizer(path):
    with open(Path(path) / "args.json", "r") as f:
        args = json.load(f)
    tokenizer, idx_w = simple_tokenizers.get_tokenizer(args["data_dir"])
    model = models.GPTNoPE.from_pretrained(path).to(torch.device(args["device"]))
    return model, args, tokenizer, idx_w


def get_attention_embeddings(model, input_ids, layer):
    model.eval()
    block = model.transformer.h[layer]
    attn = block.attn
    with torch.no_grad():
        out = model(input_ids=input_ids, return_dict=True, output_hidden_states=True)
        h = out.hidden_states[layer - 1]
        h_ = block.ln_1(h)
        query, key, value = attn.c_attn(h_).split(attn.split_size, dim=2)
        query = attn._split_heads(query, attn.num_heads, attn.head_dim).cpu().numpy()
        key = attn._split_heads(key, attn.num_heads, attn.head_dim).cpu().numpy()
        value = attn._split_heads(value, attn.num_heads, attn.head_dim).cpu().numpy()
    return query, key, value


def get_offsets(copy_groups):
    offsets = []
    prev = None
    for g in copy_groups:
        if g != prev:
            offsets.append(0)
            prev = g
        else:
            offsets.append(offsets[-1] + 1)
    return offsets


def get_full_offsets(copy_groups, template):
    offsets = []
    prev = None
    for g in copy_groups:
        # Only reset if previous state is 0
        if prev is None or (g != prev and template[prev] == "0"):
            prev = g
            offsets.append(0)
        else:
            prev = g
            offsets.append(offsets[-1] + 1)
    return offsets


def run_attention_probe(model, eval_df, batch_size, layer=-1):
    data_loader = DataLoader(
        list(zip(eval_df["input_ids"], eval_df.index)),
        batch_size=batch_size,
        shuffle=False,
    )
    query_rows, key_rows = [], []
    query_embs, key_embs = [], []
    num_batches = len(eval_df) // batch_size
    t = tqdm(data_loader, desc="eval", total=num_batches)
    model.eval()
    for batch in t:
        input_ids = batch[0].to(model.device)
        queries, keys, _ = get_attention_embeddings(model, input_ids, layer)
        idxs = batch[1].cpu().numpy()
        batch_df = eval_df.loc[idxs]
        # For each sentence
        for i, (_, row) in enumerate(batch_df.iterrows()):
            template = tuple(row["templates"].split())
            transformation = tuple(row["transformations"].split())
            s = row["input"]
            words = s.split()
            delim = [j for j, w in enumerate(words) if w == "."][0]

            # Parse the input
            sentence_idxs = list(range(1, delim + 1))
            sentence = [words[j] for j in sentence_idxs]
            states = generate_data.get_fsa_states(template, sentence)
            copy_groups = generate_data.get_copy_groups(template, states)
            offsets = get_offsets(copy_groups)
            full_offsets = get_full_offsets(copy_groups, template)
            for j, g, o, f in zip(sentence_idxs, copy_groups, offsets, full_offsets):
                # copy_prefix: The prefix restricted to copied characters with copy_group == g
                # full_copy_prefix: copy_prefix, but also including literal characters.
                #   For example, given template = 0 a b 0 c and input d a b e f g c, the copy
                #   prefixes for group 4 are "e", "e f", and " e f g", and the full copy prefixes
                #   are "a b e", "a b e f", "a b e f g".
                key_rows.append(
                    {
                        "conv_id": row["conv_id"],
                        "idx": j,
                        "word": words[j],
                        "prefix": " ".join(words[1:j]) or "-",
                        "copy_prefix": " ".join(words[j - o : j]) or "-",
                        "full_copy_prefix": " ".join(words[j - f : j]) or "-",
                        "template": row["templates"],
                        "copy_group": g,
                        "is_copy": template[g] == "0",
                        "copy_idx": o,
                    }
                )
                key_embs.append(keys[i, :, j])

            # Parse the response
            response_idxs = list(range(delim, len(words)))
            actions = []
            tgt_idxs = []
            for t in transformation:
                if type(t) == int or t.isnumeric():
                    actions += [s for s in copy_groups if s == int(t)]
                    tgt_idxs += [
                        j for j, s in zip(sentence_idxs, copy_groups) if s == int(t)
                    ]
                else:
                    actions.append(t)
                    tgt_idxs.append(-1)
            for j, a, t in zip(response_idxs, actions, tgt_idxs):
                tgt_wd = words[j + 1] if j + 1 < len(words) else ""
                o = offsets[t]
                f = full_offsets[t]
                query_rows.append(
                    {
                        "conv_id": row["conv_id"],
                        "idx": j,
                        "word": words[j],
                        "prefix": " ".join(words[delim + 1 : j + 1]) or "-",
                        "copy_prefix": " ".join(words[t - o + 1 : t + 1]) or "-",
                        "full_copy_prefix": " ".join(words[t - f + 1 : t + 1]) or "-",
                        "tgt_wd": tgt_wd,
                        "template": row["templates"],
                        "copy_group": a,
                        "is_copy": type(a) == int,
                        "tgt_idx": t,
                        "tgt_copy_idx": o,
                    }
                )
                query_embs.append(queries[i, :, j])

    query_df = pd.DataFrame(query_rows)
    key_df = pd.DataFrame(key_rows)
    return query_df, key_df, np.stack(query_embs, 0), np.stack(key_embs, 0)


def get_score_contrasts(
    query_df,
    key_df,
    query_embs,
    key_embs,
    max_rows=1024,
    ngrams=[0, 1, 2, 3, 4],
    subtract_baseline=False,
):
    # Given key and query embeddings (from across examples), calculate the average score when:
    #   - Key position == query target position, but key prefix != query prefix
    #   - Key prefix == query prefix, but key position != query target position
    lst = []
    for n in ngrams:
        key_df[f"{n+1}-gram"] = [c[-(2 * n) - 1 :] for c in key_df["full_copy_prefix"]]
        query_df[f"{n+1}-gram"] = [
            c[-(2 * n) - 1 :] for c in query_df["full_copy_prefix"]
        ]
    for i, row in tqdm(list(query_df.query("is_copy").iterrows())[:max_rows]):
        q = query_embs[i]
        # same n-gram
        for n in ngrams:
            ngram = row[f"{n+1}-gram"]
            ngram_mask = key_df[f"{n+1}-gram"].to_numpy() == ngram
            pos_mask = row["tgt_idx"] == key_df["idx"].to_numpy()
            same_diff = ngram_mask & (~pos_mask)
            diff_same = (~ngram_mask) & pos_mask
            if same_diff.sum() == 0 or diff_same.sum() == 0:
                continue
            baseline = 0
            baseline_mask = (~ngram_mask) & (~pos_mask)
            if subtract_baseline:
                baseline = (q * key_embs[baseline_mask].mean(0)).sum(-1)
            for metric, mask in (
                (f"same_ngram_diff_pos", same_diff),
                (f"diff_ngram_same_pos", diff_same),
                (f"diff_ngram_diff_pos", baseline_mask),
            ):
                if subtract_baseline and metric == "diff_ngram_diff_pos":
                    continue
                k = key_embs[mask].mean(0)
                lst.append(
                    {
                        "idx": i,
                        "condition": metric,
                        "n": n,
                        "score": (q * k).sum(-1) - baseline,
                    }
                )
    by_head = []
    for row in lst:
        for h, s in enumerate(row["score"]):
            r = copy.deepcopy(row)
            r["head"] = h
            r["score"] = s
            by_head.append(r)
    score_df = pd.DataFrame(by_head)
    return score_df


def run(args):
    logger.info(f"Args: {vars(args)}")

    logger.info(f"Loading model...")
    model, model_args, tokenizer, _ = load_model_and_tokenizer(args.model_dir)
    if args.data_dir is None:
        args.data_dir = model_args["data_dir"]

    logger.info(f"Loading data...")
    val_df = data_utils.load_dataset(Path(args.data_dir) / "validation.csv", tokenizer)
    if args.max_examples:
        val_df = val_df.head(args.max_examples)

    query_df, key_df, query_embs, key_embs = run_attention_probe(
        model, val_df, batch_size=args.batch_size, layer=args.attention_layer
    )

    fn = Path(args.output_dir) / "queries.csv"
    logger.info(f"Writing {len(query_df)} queries to {fn}")
    query_df.to_csv(fn)

    fn = Path(args.output_dir) / "keys.csv"
    logger.info(f"Writing {len(key_df)} queries to {fn}")
    key_df.to_csv(fn)

    fn = Path(args.output_dir) / "queries.npy"
    logger.info(f"Writing {query_embs.shape} query embeddings to {fn}")
    np.save(fn, query_embs)

    fn = Path(args.output_dir) / "keys.npy"
    logger.info(f"Writing {key_embs.shape} key embeddings to {fn}")
    np.save(fn, key_embs)

    logger.info(f"Getting score contrasts")
    score_df = get_score_contrasts(
        query_df, key_df, query_embs, key_embs, max_rows=args.max_rows
    )
    fn = Path(args.output_dir) / "scores.csv"
    logger.info(f"Writing {len(score_df)} rows to {fn}")
    score_df.to_csv(fn)


if __name__ == "__main__":
    args = parse_args()
    logging.initialize(args.output_dir)
    set_seed(args.seed)
    run(args)
