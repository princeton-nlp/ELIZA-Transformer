import numpy as np
import pandas as pd


def pad_lst(seq, max_len, value=0):
    return seq + [value] * (max_len - len(seq))


def load_dataset(fn, tokenizer):
    df = pd.read_csv(fn)
    encs = tokenizer.encode_batch(df["input"])
    rows = []
    max_len = max(len(enc.ids) for enc in encs)
    for mask, enc in zip(df["mask"], encs):
        # Labels are shifted inside HF GPT2
        input_ids = enc.ids
        labels = [-100 if m == "0" else t for m, t in zip(mask.split(" "), enc.ids)]
        rows.append(
            {
                "input_ids": np.array(pad_lst(input_ids, max_len, 0)),
                "labels": np.array(pad_lst(labels, max_len, -100)),
            }
        )
    df["input_ids"] = [r["input_ids"] for r in rows]
    df["labels"] = [r["labels"] for r in rows]
    return df


def split_conversations(df: pd.DataFrame):
    out = []
    for _, row in df.iterrows():
        inputs = ["U" + s for s in row["input"].split("U")[1:]]
        masks = []
        i = 0
        for s in inputs:
            masks.append(row["mask"][i : i + len(s)])
            i += len(s)
        templates = str(row["templates"]).split(";")
        template_ids = str(row["template_ids"]).split(";")
        transformations = str(row["transformations"]).split(";")
        transformation_ids = str(row["transformation_ids"]).split(";")
        if "turn_types" in row:
            turn_types = str(row["turn_types"]).split(";")
        else:
            turn_types = ["-"] * len(templates)
        conv_id = row["conv_id"]
        turn = 0
        for (
            s,
            mask,
            template,
            template_id,
            transformation,
            transformation_id,
            turn_type,
        ) in zip(
            inputs,
            masks,
            templates,
            template_ids,
            transformations,
            transformation_ids,
            turn_types,
        ):
            sent = s[: s.find("E")]
            response = s[s.find("E") :]
            out.append(
                {
                    "conv_id": conv_id,
                    "turn": turn,
                    "template": template,
                    "transformation": transformation,
                    "sentence": sent,
                    "response": response,
                    "input": s,
                    "mask": mask,
                    "template_id": template_id,
                    "transformation_id": transformation_id,
                    "turn_type": turn_type,
                }
            )
            for k in ["path"]:
                if k in row:
                    out[-1][k] = row[k]
            turn += 1
    return pd.DataFrame(out)
