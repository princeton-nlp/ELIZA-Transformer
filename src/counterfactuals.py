import argparse
from pathlib import Path
import random
import shutil

import pandas as pd
import numpy as np
from tqdm import tqdm

from src import generate_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/combined/10k")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--edit_type", type=str, default="cycling", choices=["cycling", "memory_queue"]
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.data_dir + "_counterfactual"
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def process_script(script):
    templates = [""] * (script["template_id"].max() + 1)
    transformations = [[] for _ in range(len(templates))]
    memory_template_id = None
    memory_transformations = []
    for _, row in script.iterrows():
        if row["type"] == "memory":
            t = tuple(row["template"].split())
            assert t in templates
            memory_template_id = templates.index(t)
            memory_transformations.append(tuple(row["transformation"].split()))
            continue
        templates[row["template_id"]] = tuple(row["template"].split())
        transformations[row["template_id"]].append(tuple(row["transformation"].split()))
    return templates, transformations, memory_template_id, memory_transformations


def do_cycling_edit(row, transformations, memory_template_id):
    # Remove last turn to ensure the new conversation isn't too long.
    conv_template_ids = [int(t) for t in row["template_ids"].split(";")[:-1]]
    conv_templates = row["templates"].split(";")[:-1]
    t_id, template = conv_template_ids[-1], conv_templates[-1]

    transform_lst = transformations[t_id]
    if (
        t_id == memory_template_id
        or template == "U 0 ."
        or t_id not in conv_template_ids[:-1]
        or len(transform_lst) <= 1
    ):
        return None, None

    template = tuple(template.split())
    turns = ["U" + s for s in row["input"].split("U")[1:]][:-1]
    turns[0] = "$ " + turns[0]
    turn_types = row["turn_types"].split(";")[:-1]

    def rindex(lst, elem):
        return len(lst) - 1 - lst[::-1].index(elem)

    # Find the most recent instance of the template
    prev_idx = rindex(conv_template_ids[:-1], t_id)
    prev_count = sum(t_ == t_id for t_ in conv_template_ids[:prev_idx])
    prev_transformation_id = prev_count % len(transform_lst)

    # Change the previous cycle number
    diff = random.randint(1, len(transform_lst) - 1)
    new_prev_transformation_id = (prev_transformation_id + diff) % len(transform_lst)
    offset = prev_transformation_id - new_prev_transformation_id
    new_prev_transformation = transform_lst[new_prev_transformation_id]

    # Update the previous turn
    prev_words = turns[prev_idx].split()
    r_idx = prev_words.index("E")
    prev_sentence = prev_words[:r_idx]
    _, prev_response = generate_data.do_transformation(
        template, new_prev_transformation, prev_sentence[int(prev_sentence[0] == "$") :]
    )
    new_prev_turn = list(map(str, prev_sentence + prev_response))

    # Update the final turn
    new_transformation_id = (new_prev_transformation_id + 1) % len(transform_lst)
    new_transformation = transform_lst[new_transformation_id]
    words = turns[-1].split()
    r_idx = words.index("E")
    sentence = words[:r_idx]
    _, response = generate_data.do_transformation(
        template, new_transformation, sentence
    )
    new_turn = list(map(str, sentence + response))

    def make_new_mask(s):
        cur = "0"
        new_mask = []
        for w in s.split():
            if w == "E":
                new_mask.append("0")
                cur = "1"
            elif w == "U":
                new_mask.append("0")
                cur = "0"
            else:
                new_mask.append(cur)
        return " ".join(new_mask)

    # Generate two new versions (same prediction, changed prediction)
    # 1. Change previous response, keep same prediction
    out_v1 = row.to_dict()
    new_input = " ".join(
        [
            "".join(turns[:prev_idx])
            + " ".join(new_prev_turn)
            + "".join(turns[prev_idx + 1 :])
        ]
    )
    if len(new_input.split()) > 512:
        return None, None
    out_v1["input"] = new_input
    out_v1["mask"] = make_new_mask(new_input)
    out_v1["turn_types"] = ";".join(turn_types[:-1] + ["same_transformation"])
    out_v1["edit_type"] = "same_transformation"
    out_v1["diff"] = offset
    out_v1["template_count"] = prev_count + 1
    out_v1["cycle_number"] = (prev_count + 1) // len(transform_lst)
    for k in ["templates", "template_ids", "transformations", "transformation_ids"]:
        out_v1[k] = ";".join(out_v1[k].split(";")[:-1])

    # 2. Change previous response, change current prediction
    out_v2 = row.to_dict()
    new_input = " ".join(
        [
            "".join(turns[:prev_idx])
            + " ".join(new_prev_turn)
            + "".join(turns[prev_idx + 1 : -1])
            + " ".join(new_turn)
        ]
    )
    if len(new_input.split()) > 512:
        return None, None
    out_v2["input"] = new_input
    out_v2["mask"] = make_new_mask(new_input)
    out_v2["turn_types"] = ";".join(turn_types[:-1] + ["increment_transformation"])
    out_v2["edit_type"] = "increment_transformation"
    out_v2["diff"] = offset
    out_v2["template_count"] = prev_count + 1
    out_v2["cycle_number"] = (prev_count + 1) // len(transform_lst)
    for k in ["templates", "template_ids", "transformations", "transformation_ids"]:
        out_v2[k] = ";".join(out_v2[k].split(";")[:-1])

    return out_v1, out_v2


def do_memory_queue_edit(row, transformations, memory_template_id):
    del memory_template_id

    # Remove last of everything so it isn't too long.
    turn_types = row["turn_types"].split(";")[:-1]
    if sum(t == "dequeue" for t in turn_types) < 2:
        return None, None

    def rindex(lst, elem):
        return len(lst) - 1 - lst[::-1].index(elem)

    last_idx = rindex(turn_types, "dequeue")
    prev_idx = rindex(turn_types[:last_idx], "dequeue")

    conv_template_ids = [int(t) for t in row["template_ids"].split(";")[: last_idx + 1]]
    conv_templates = row["templates"].split(";")[: last_idx + 1]
    t_id, template = conv_template_ids[-1], conv_templates[-1]

    template = tuple(template.split())
    turns = ["U" + s for s in row["input"].split("U")[1:]][: last_idx + 1]
    turns[0] = "$ " + turns[0]
    # turns = [turns[0]] + [". U" + t for t in turns[1:]]

    # This is the null transformation...
    new_prev_transformation = transformations[-2][0]

    # Find the most recent dequeue and change it to a null response.
    prev_words = turns[prev_idx].split()
    r_idx = prev_words.index("E")
    prev_sentence = prev_words[:r_idx]
    _, prev_response = generate_data.do_transformation(
        template, new_prev_transformation, prev_sentence[int(prev_sentence[0] == "$") :]
    )
    new_prev_turn = list(map(str, prev_sentence + prev_response))

    # This will be the expected response for the final dequeue
    old_prev_response = prev_words[r_idx:]

    # Update the final turn
    words = turns[-1].split()
    r_idx = words.index("E")
    sentence = words[:r_idx]
    response = old_prev_response
    new_turn = list(map(str, sentence + response))

    def make_new_mask(s):
        cur = "0"
        new_mask = []
        for w in s.split():
            if w == "E":
                new_mask.append("0")
                cur = "1"
            elif w == "U":
                new_mask.append("0")
                cur = "0"
            else:
                new_mask.append(cur)
        return " ".join(new_mask)

    # Generate two new versions (same prediction, changed prediction)
    # 1. Change previous response, keep same prediction
    out_v1 = row.to_dict()
    new_input = " ".join(
        [
            "".join(turns[:prev_idx])
            + " ".join(new_prev_turn)
            + "".join(turns[prev_idx + 1 :])
        ]
    )
    if len(new_input.split()) > 512:
        return None, None
    out_v1["input"] = new_input
    out_v1["mask"] = make_new_mask(new_input)
    out_v1["turn_types"] = ";".join(turn_types[:last_idx] + ["same_dequeue"])
    out_v1["edit_type"] = "same_dequeue"
    for k in ["templates", "template_ids", "transformations", "transformation_ids"]:
        out_v1[k] = ";".join(out_v1[k].split(";")[: last_idx + 1])

    # 2. Change previous response, change current prediction
    out_v2 = row.to_dict()
    new_input = " ".join(
        [
            "".join(turns[:prev_idx])
            + " ".join(new_prev_turn)
            + "".join(turns[prev_idx + 1 : -1])
            + " ".join(new_turn)
        ]
    )
    if len(new_input.split()) > 512:
        return None, None
    out_v2["input"] = new_input
    out_v2["mask"] = make_new_mask(new_input)
    out_v2["turn_types"] = ";".join(turn_types[:last_idx] + ["decrement_dequeue"])
    out_v2["edit_type"] = "decrement_dequeue"
    for k in ["templates", "template_ids", "transformations", "transformation_ids"]:
        out_v2[k] = ";".join(out_v2[k].split(";")[: last_idx + 1])

    return out_v1, out_v2


def generate_counterfactuals(args):
    data_dir = Path(args.data_dir)
    script = pd.read_csv(data_dir / "script.csv")
    val = pd.read_csv(data_dir / "validation.csv")
    _, transformations, memory_template_id, _ = process_script(script)
    out = []
    for _, row in tqdm(val.iterrows(), total=len(val)):
        if args.edit_type == "cycling":
            a, b = do_cycling_edit(row, transformations, memory_template_id)
        elif args.edit_type == "memory_queue":
            a, b = do_memory_queue_edit(row, transformations, memory_template_id)
        else:
            raise NotImplementedError(args.edit_type)
        if a is not None:
            out += [a, b]
    return pd.DataFrame(out)


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    new_val = generate_counterfactuals(args)
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    print(f"Writing {len(new_val)//2} new examples to {output_dir}")
    new_val.to_csv(output_dir / "validation.csv")
    shutil.copy(
        Path(args.data_dir) / "tokenizer.json", Path(args.output_dir) / "tokenizer.json"
    )
