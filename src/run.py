import argparse
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

from src import data_utils, models, simple_tokenizers
from src.utils import logging

logger = logging.get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="output/scratch")
    parser.add_argument(
        "--data_dir", type=str, default="data/single_turn/templates32_tlen10_slen32"
    )
    parser.add_argument("--eval_on", type=str, nargs="*")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--load_from", type=str, default=None)

    # Model configuration
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--hidden_size", type=int, default=768)

    # Training configuration
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--save", action="store_true")

    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_training_epoch(model, opt, train_df, batch_size, max_grad_norm=None):
    data_loader = DataLoader(
        list(zip(train_df["input_ids"], train_df["labels"])),
        batch_size=batch_size,
        shuffle=True,
    )
    losses = []
    num_batches = len(train_df) // batch_size
    t = tqdm(data_loader, desc="train", total=num_batches)
    model.train()
    for batch in t:
        input_ids = batch[0].to(model.device)
        labels = batch[1].to(model.device)
        loss = model(input_ids=input_ids, labels=labels, return_dict=True).loss
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_grad_norm
            )
        losses.append(loss.item())
        opt.step()
        opt.zero_grad()
        model.zero_grad()
        t.set_postfix({"loss": losses[-1]})
    return losses


def run_eval(model, eval_df, idx_w, batch_size):
    data_loader = DataLoader(
        list(zip(eval_df["input_ids"], eval_df["labels"])),
        batch_size=batch_size,
        shuffle=False,
    )
    rows = []
    eval_losses = []
    num_batches = len(eval_df) // batch_size
    t = tqdm(data_loader, desc="eval", total=num_batches)
    model.eval()
    for batch in t:
        input_ids = batch[0].to(model.device)
        labels_ = batch[1].to(model.device)
        with torch.no_grad():
            out = model(input_ids=input_ids, labels=labels_, return_dict=True)
            logits = out.logits[:, :-1]
            labels = labels_[:, 1:]
            eval_losses.append(out.loss.item())

            # Loss
            log_probs = logits.log_softmax(-1)
            tgts = F.relu(labels)
            all_losses = -log_probs.gather(2, tgts.unsqueeze(-1)).squeeze(-1)
            masked_losses = all_losses.masked_fill((labels == -100), 0.0)
            lengths = (labels != -100).sum(-1)
            losses = (masked_losses.sum(-1) / lengths).cpu().numpy()
            loss = losses.mean().item()

            # Predictions
            preds = idx_w[log_probs.argmax(-1).detach().cpu().numpy()]
            tgt_wds = idx_w[tgts.cpu().numpy()]
            mask = (labels != -100).cpu().numpy()

            # Split conversation into turns
            for pred_, tgt_, m_, ls_ in zip(
                preds, tgt_wds, mask, masked_losses.cpu().numpy()
            ):
                starts = [0] + [i + 1 for i, w in enumerate(tgt_) if w == "."]
                for turn, (s, e) in enumerate(zip(starts[:-1], starts[1:])):
                    pred, tgt, m, ls = (pred_[s:e], tgt_[s:e], m_[s:e], ls_[s:e])
                    want = " ".join(tgt[m])
                    got = " ".join(pred[m])
                    want_prefix = " ".join(tgt[m][:2])
                    got_prefix = " ".join(pred[m][:2])
                    l = ls.sum() / m.sum()
                    rows.append(
                        {
                            "turn": turn,
                            "want": want,
                            "got": got,
                            "loss": l,
                            "acc": want == got,
                            "prefix_acc": want_prefix == got_prefix,
                        }
                    )

        t.set_postfix({"loss": loss})

    results = pd.DataFrame(rows)
    turn_df = data_utils.split_conversations(eval_df)
    assert len(results) == len(turn_df)
    for k in [
        "conv_id",
        "template_id",
        "transformation_id",
        "template",
        "transformation",
        "turn_type",
        "input",
        "path",
    ]:
        if k in turn_df:
            results[k] = turn_df[k]
    return eval_losses, results


def split_training_set(df, eval_every, batch_size):
    idxs = np.arange(len(df))
    chunk_size = eval_every * batch_size
    num_chunks = len(df) // chunk_size
    np.random.shuffle(idxs)
    lst = []
    for i in range(num_chunks):
        lst.append(df.iloc[idxs[i * chunk_size : (i + 1) * chunk_size]])
    return lst


def load_model_and_tokenizer(path):
    with open(path / "args.json", "r") as f:
        args = json.load(f)
    tokenizer, idx_w = simple_tokenizers.get_tokenizer(args["data_dir"])
    model = models.GPTNoPE.from_pretrained(path).to(torch.device(args["device"]))
    return model, tokenizer, idx_w


def run_eval_only(model, val_df, idx_w, args, additional_eval_df=None):
    eval_losses, predictions = run_eval(model, val_df, idx_w, args.eval_batch_size)
    eval_loss = np.mean(eval_losses)
    logger.info(
        f"Val loss: {eval_loss} "
        f"Val acc: {predictions['acc'].mean()} "
        f"Val prefix acc: {predictions['prefix_acc'].mean()}"
    )
    output_dir = Path(args.output_dir)
    predictions.to_csv(output_dir / f"predictions.csv")
    metrics = predictions[["loss", "acc", "prefix_acc"]].mean().reset_index()
    metrics.to_csv(output_dir / "metrics.csv")
    metrics_by_turn = (
        predictions.groupby(["turn_type"])[["loss", "acc", "prefix_acc"]]
        .mean()
        .reset_index()
    )
    metrics_by_turn.to_csv(output_dir / "metrics_by_turn.csv")

    if additional_eval_df is not None:
        _, additional_predictions = run_eval(
            model, additional_eval_df, idx_w, args.eval_batch_size
        )
        acc = additional_predictions.groupby(["path"])[["loss", "acc"]].mean()
        logger.info(f"Additional eval: {acc}")
        additional_predictions.to_csv(output_dir / f"additional_predictions.csv")
        additional_metrics = (
            additional_predictions.groupby(["epoch", "path"])[["loss", "acc"]]
            .mean()
            .reset_index()
        )
        additional_metrics.to_csv(output_dir / "additional_metrics.csv")


def run(args):
    logger.info(f"Args: {vars(args)}")

    logger.info(f"Loading data...")
    tokenizer, idx_w = simple_tokenizers.get_tokenizer(args.data_dir)
    train_df = None
    if not args.eval_only:
        train_df = data_utils.load_dataset(Path(args.data_dir) / "train.csv", tokenizer)
    val_df = data_utils.load_dataset(Path(args.data_dir) / "validation.csv", tokenizer)

    additional_eval_df = None
    if args.eval_on:
        lst = []
        for path in args.eval_on:
            df = data_utils.load_dataset(Path(path) / "validation.csv", tokenizer)
            df["path"] = path
            lst.append(df)
        additional_eval_df = pd.concat(lst)

    if args.load_from:
        logger.info(f"Loading model from {args.load_from}")
        model, _, _ = load_model_and_tokenizer(Path(args.load_from))
    else:
        logger.info(f"Loading model...")
        config = GPT2Config(
            vocab_size=tokenizer.get_vocab_size(),
            n_embd=args.hidden_size,
            n_layer=args.num_layers,
            n_head=args.num_heads,
        )
        model = models.GPTNoPE(config).to(torch.device(args.device))
    opt = Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    output_dir = Path(args.output_dir)
    logger.info(f"Writing results to {output_dir}")
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    if args.eval_only:
        return run_eval_only(
            model, val_df, idx_w, args, additional_eval_df=additional_eval_df
        )

    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f)

    best_loss = 1e10
    train_losses = []
    for epoch in range(args.num_epochs):
        logger.info(f"Epoch {epoch}")
        if args.eval_every is None:
            dfs = [train_df]
        else:
            dfs = split_training_set(train_df, args.eval_every, args.train_batch_size)
        for i, df in enumerate(dfs):
            do_header = (epoch == 0) & (i == 0)
            losses = run_training_epoch(
                model,
                opt,
                df,
                args.train_batch_size,
                max_grad_norm=args.max_grad_norm,
            )
            steps = np.arange(len(losses)) + len(train_losses)
            step = steps[-1]
            loss_df = pd.DataFrame({"epoch": epoch, "step": steps, "loss": losses})
            loss_df.to_csv(output_dir / "train_loss.csv", mode="a", header=do_header)
            train_losses += losses
            logger.info(
                f"Epoch {epoch}, step {step}. Avg. train loss: {np.mean(losses)}"
            )
            eval_losses, predictions = run_eval(
                model, val_df, idx_w, args.eval_batch_size
            )
            eval_loss = np.mean(eval_losses)
            logger.info(
                f"Epoch {epoch}, step {step}. "
                f"Val loss: {eval_loss} "
                f"Val acc: {predictions['acc'].mean()} "
                f"Val prefix acc: {predictions['prefix_acc'].mean()}"
            )
            predictions["epoch"] = epoch
            predictions["step"] = step
            predictions.to_csv(output_dir / f"predictions_{epoch:02d}_{step}.csv")
            metrics = (
                predictions.groupby(["epoch", "step"])[["loss", "acc", "prefix_acc"]]
                .mean()
                .reset_index()
            )
            metrics.to_csv(output_dir / "metrics.csv", mode="a", header=do_header)
            metrics_by_turn = (
                predictions.groupby(["epoch", "step", "turn_type"])[
                    ["loss", "acc", "prefix_acc"]
                ]
                .mean()
                .reset_index()
            )
            metrics_by_turn.to_csv(
                output_dir / "metrics_by_turn.csv", mode="a", header=do_header
            )

            if eval_loss < best_loss:
                logger.info(f"New best loss at epoch {epoch}: {eval_loss}")
                best_loss = eval_loss
                if args.save:
                    logger.info(f"Saving model...")
                    model.save_pretrained(str(output_dir), from_pt=True)

            if additional_eval_df is not None:
                _, additional_predictions = run_eval(
                    model, additional_eval_df, idx_w, args.eval_batch_size
                )
                acc = additional_predictions.groupby(["path"])[["loss", "acc"]].mean()
                logger.info(f"Additional eval: {acc}")
                additional_predictions["epoch"] = epoch
                additional_predictions.to_csv(
                    output_dir / f"additional_predictions_{epoch:02d}.csv"
                )
                additional_metrics = (
                    additional_predictions.groupby(["epoch", "path"])[["loss", "acc"]]
                    .mean()
                    .reset_index()
                )
                additional_metrics.to_csv(
                    output_dir / "additional_metrics.csv", mode="a", header=do_header
                )


if __name__ == "__main__":
    args = parse_args()
    logging.initialize(args.output_dir)
    set_seed(args.seed)
    run(args)
