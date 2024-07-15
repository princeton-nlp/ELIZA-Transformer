import argparse
from collections import deque
import json
from pathlib import Path
import random
import string

import numpy as np
import pandas as pd
from scipy.stats import dirichlet
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/scratch/")
    parser.add_argument("--seed", type=int, default=0)

    # Script
    parser.add_argument("--num_templates", type=int, default=32)
    parser.add_argument("--max_transformations_per_template", type=int, default=1)
    parser.add_argument("--null_template", action="store_true")
    parser.add_argument("--memory_queue", action="store_true")
    parser.add_argument("--max_queue_size", type=int, default=4)
    parser.add_argument("--memory_ratio", type=int, default=1)
    parser.add_argument("--null_template_cycle_version", type=int, default=2)

    # Template
    parser.add_argument("--max_template_len", type=int, default=12)
    parser.add_argument("--min_num_wildcards", type=int, default=2)
    parser.add_argument("--max_num_wildcards", type=int, default=4)
    parser.add_argument("--max_ngram_len", type=int, default=3, help="Not used")

    # Transformation
    parser.add_argument("--template_prefix_len", type=int, default=2)
    parser.add_argument("--min_copies", type=int, default=1)
    parser.add_argument("--max_copies", type=int, default=4)

    # Dataset
    parser.add_argument("--num_conversations", type=int, default=4096)
    parser.add_argument(
        "--examples_per_template", type=int, default=2048, help="Not used"
    )

    # Conversation
    parser.add_argument("--max_turns", type=int, default=1)
    parser.add_argument("--max_conversation_len", type=int, default=512)
    parser.add_argument("--concentration", type=float, default=1.0)
    parser.add_argument("--no_bos", action="store_true")

    # Turn
    parser.add_argument("--min_copy_len", type=int, default=0)
    parser.add_argument("--max_copy_len", type=int, default=10)
    parser.add_argument("--unigram_concentration", type=float, default=100.0)
    parser.add_argument("--max_sentence_len", type=int, default=32, help="Not used")

    args = parser.parse_args()
    if args.memory_queue:
        assert args.null_template
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


class FSA:
    def __init__(self, graph):
        self.graph = graph
        self.edges = {k: dict(v) for k, v in graph.items()}

    def sample(self, init=0, max_len=10):
        state = init
        states, words = [state], ["U"]
        while state in self.graph and len(words) < max_len:
            word, state = random.choice(self.graph[state])
            states.append(state)
            words.append(word)
        return states, words

    def get_states(self, sentence, init=0):
        state = init
        states = [init]
        for w in sentence[1:]:
            if w not in self.edges[state]:
                return None
            state = self.edges[state][w]
            states.append(state)
        return states


def sample_template(
    min_num_wildcards=1,
    max_num_wildcards=4,
    max_ngram_len=3,
    vocab=string.ascii_lowercase,
    **kwargs,
):
    num_wildcards = random.randint(min_num_wildcards, max_num_wildcards)
    out = ["U"]
    for i in range(num_wildcards):
        if i == 0:
            out += random.choices(vocab, k=random.randint(0, max_ngram_len))
        else:
            out += random.choices(vocab, k=random.randint(1, max_ngram_len))
        out.append("0")
    out += random.choices(vocab, k=random.randint(0, max_ngram_len))
    out.append(".")
    return out


def get_graph_for_template(template, vocab=string.ascii_lowercase):
    graph = {}
    template_str = "".join(template)
    last_0 = None
    for i in range(len(template) - 1):
        t = template[i]
        t_next = template[i + 1]
        if t_next == "0":
            assert i + 2 < len(template)
            t_skip = template[i + 2]
            graph[i] = [(t_skip, i + 2)] * len(vocab) + [
                (c, i + 1) for c in vocab if c != t_skip
            ]
        elif t == "0":
            last_0 = i
            graph[i] = [(t_next, i + 1)] * len(vocab) + [
                (c, i) for c in vocab if c != t_next
            ]
        elif last_0 is not None:
            # Possibility of returning to the last wildcard state.
            # If letter is anything other than t_next, transition as
            # if we were still in the last wildcard state.
            graph[i] = [(t_next, i + 1)] * len(vocab) + [
                e for e in graph[last_0] if e[0] != t_next
            ]
        else:
            graph[i] = [(t_next, i + 1)]
    fsa = FSA(graph)
    return fsa


def split_template(template):
    parts = []
    part = []
    for t in template:
        if t == "0":
            parts += [tuple(part), "0"]
            part = []
        elif t != ".":
            part.append(t)
        else:
            if part:
                parts.append(tuple(part))
            parts.append((".",))
    return parts


def is_subseq(x, y):
    it = iter(y)
    return all(any(c == ch for c in it) for ch in x)


def sample_part(
    vocab=string.ascii_lowercase, min_len=0, max_len=16, unigram_concentration=100.0
):
    l = random.randint(min_len, max_len)
    if l == 0:
        return []
    if unigram_concentration < 0:
        p = np.ones(len(vocab)) / len(vocab)
    else:
        p = dirichlet.rvs(unigram_concentration * np.ones(len(vocab)))[0]
    return random.choices(vocab, weights=p, k=l)


def sample_sentence_for_template(
    template,
    vocab=string.ascii_lowercase,
    min_copy_len=0,
    max_copy_len=10,
    unigram_concentration=1.0,
):
    out = []
    states = []
    parts = split_template(template)
    for i, part in enumerate(parts):
        if part != "0":
            out += list(part)
            states += [i] * len(part)
            continue
        min_len = 1 if is_null_template(template) else min_copy_len
        max_len = max(1, max_copy_len) if is_null_template else max_copy_len
        s = tuple(
            sample_part(
                vocab=vocab,
                min_len=min_len,
                max_len=max_len,
                unigram_concentration=unigram_concentration,
            )
        )
        # Make sure the next n-gram isn't part of this segment.
        while i < len(parts) - 1 and is_subseq(parts[i + 1], s):
            s = tuple(
                sample_part(
                    vocab=vocab,
                    min_len=min_len,
                    max_len=max_len,
                    unigram_concentration=unigram_concentration,
                )
            )
        out += list(s)
        states += [i] * len(s)
    return states, out


def sample_transformation_for_template(
    template,
    vocab=string.ascii_lowercase,
    max_ngram_len=3,
    min_copies=0,
    max_copies=10,
    template_prefix_len=2,
    seen_prefixes=None,
    **kwargs,
):
    capture_groups = [i for i, t in enumerate(template) if t == "0"]
    num_copies = random.randint(
        min(len(capture_groups), min_copies),
        min(len(capture_groups), max_copies),
    )
    prefix = []
    if template_prefix_len > 0:
        prefix = random.choices(vocab, k=template_prefix_len)
        while tuple(prefix) in seen_prefixes:
            prefix = random.choices(vocab, k=template_prefix_len)
    copies = []
    if num_copies > 0:
        copies = random.sample(capture_groups, num_copies)
    out = ["E"] + prefix
    for c in copies:
        out += random.choices(vocab, k=random.randint(0, max_ngram_len))
        out.append(c)
    out += random.choices(vocab, k=random.randint(0, max_ngram_len))
    out.append(".")
    return out


def get_copy_groups(template, states):
    out = []
    last_index = {s: i for i, s in enumerate(states)}
    template_str = "".join(template)
    for i, s in enumerate(states):
        if template[s] == "0":
            out.append(s)
            continue
        # If this state appears anywhere to the right, it means
        # this is actually part of the wildcard state.
        last_0 = template_str[:s].rfind("0")
        if last_0 != -1 and i != last_index[s]:
            out.append(last_0)
        else:
            out.append(s)
    return out


def match_template(template, sentence):
    sentence = np.array(sentence)
    next_matches = [sentence == "U"]

    def just_matched(v):
        return np.concatenate([np.array([False]), v[:-1]])

    def ever_matched(v):
        return just_matched(v).cumsum() > 0

    for l in range(1, len(template)):
        if template[l] == "0":
            new_matches = ever_matched(next_matches[-1])
        elif template[l - 1] == "0":
            new_matches = next_matches[-1] & (sentence == template[l])
        else:
            new_matches = just_matched(next_matches[-1]) & (sentence == template[l])
        next_matches.append(new_matches)

    s = np.stack(next_matches)
    ind = np.arange(s.shape[0])
    states = (ind[:, None] * s).max(0)
    return s, states


def get_states(template, sentence):
    _, states = match_template(template, sentence)
    return states


def get_fsa_states(template, sentence, vocab=string.ascii_lowercase):
    fsa = get_graph_for_template(template, vocab)
    return fsa.get_states(sentence)


def do_transformation(template, transformation, sentence, vocab=string.ascii_lowercase):
    # fsa = get_graph_for_template(template, vocab)
    # states = fsa.get_states(sentence)
    states = get_states(template, sentence)
    if states is None:
        t = " ".join(template)
        r = " ".join(map(str, transformation))
        s = " ".join(sentence)
        raise ValueError(f"Couldn't parse s='{s}' with rule '{t}' -> '{r}'")
    copy_groups = get_copy_groups(template, states)
    out = []
    for t in transformation:
        if type(t) == int or t.isnumeric():
            out += [w for w, s in zip(sentence, copy_groups) if s == int(t)]
        else:
            out.append(t)
    return copy_groups, out


def find_matches(templates, sentence, vocab=string.ascii_lowercase):
    num_matches = 0
    first = -1
    for i, template in enumerate(templates):
        # fsa = get_graph_for_template(template, vocab)
        # if fsa.get_states(sentence) is not None:
        states = get_states(template, sentence)
        if states[-1] == len(template) - 1:
            num_matches += 1
            if first == -1:
                first = i
    return first, num_matches


def is_null_template(t):
    return tuple(t) == ("U", "0", ".")


def generate_script(args):
    script = []
    seen_templates = set()
    templates = []
    transformations = []
    for t in range(args.num_templates):
        template = sample_template(
            min_num_wildcards=args.min_num_wildcards,
            max_num_wildcards=args.max_num_wildcards,
            max_ngram_len=args.max_ngram_len,
            max_len=args.max_template_len,
        )
        # Make sure the template isn't equivalent to any others.
        # Any templates that are identical after removing 0s are equivalent
        # (always match the same inputs).
        template_c = tuple(c for c in template if c != "0")
        while template_c in seen_templates or len(template) > args.max_template_len:
            template = sample_template(
                min_num_wildcards=args.min_num_wildcards,
                max_num_wildcards=args.max_num_wildcards,
                max_ngram_len=args.max_ngram_len,
                max_len=args.max_template_len,
            )
            template_c = tuple(c for c in template if c != "0")
        seen_templates.add(template_c)
        templates.append(template)

    # Give longer templates higher priority.
    templates = sorted(templates, key=lambda t: len(t), reverse=True)

    if args.null_template:
        templates.append(["U", "0", "."])

    memory_template_id = None
    memory_template = None
    if args.memory_queue:
        # Pick a random template to be the memory template
        memory_template_id = random.randint(0, len(templates) - 2)
        memory_template = templates[memory_template_id]
        templates.append(memory_template)

    seen_prefixes = set()

    for t, template in enumerate(templates):
        lst = []
        n = random.randint(1, args.max_transformations_per_template)
        for r in range(n):
            transformation = sample_transformation_for_template(
                template=template,
                max_ngram_len=args.max_ngram_len,
                max_len=args.max_template_len,
                min_copies=0 if is_null_template(template) else args.min_copies,
                max_copies=0 if is_null_template(template) else args.max_copies,
                template_prefix_len=args.template_prefix_len,
                seen_prefixes=seen_prefixes,
            )
            if args.template_prefix_len:
                seen_prefixes.add(
                    tuple(transformation[1 : 1 + args.template_prefix_len])
                )
            lst.append(transformation)
            if is_null_template(template):
                rule_type = "none"
            elif args.memory_queue and t == len(templates) - 1:
                rule_type = "memory"
            else:
                rule_type = "transformation"
            script.append(
                {
                    "template_id": t,
                    "transformation_id": r,
                    "template": " ".join(template),
                    "transformation": " ".join(map(str, transformation)),
                    "template_length": len(template),
                    "transformation_length": len(transformation),
                    "num_copies": sum([type(c) == int for c in transformation]),
                    "type": rule_type,
                }
            )
        transformations.append(lst)
    if args.memory_queue:
        return (
            templates[:-1],
            transformations[:-1],
            memory_template_id,
            transformations[-1],
            script,
        )
    return templates, transformations, None, None, script


def get_max_turn_length(args):
    return (
        4
        + 2 * args.max_num_wildcards * args.max_copy_len
        + (3 + 2 * args.max_num_wildcards) * args.max_ngram_len
        + args.template_prefix_len
    )


def generate_data(args):
    templates, transformations, memory_template_id, memory_transformations, script = (
        generate_script(args)
    )
    rows = []
    # Make the memory template more likely
    alpha = args.concentration * np.ones(len(templates))
    if memory_template_id is not None:
        alpha[memory_template_id] = alpha[memory_template_id] * args.memory_ratio
    for conv_id in tqdm(range(args.num_conversations), total=args.num_conversations):
        p = dirichlet.rvs(alpha)[0]
        # Introduce correlation between memory template and null template
        # -- make sure dequeue is at least half has likely as enqueue
        if memory_template_id is not None:
            p[-1] = max(p[-1], 0.5 * p[memory_template_id])
            p = p / p.sum()
        conv_words = []
        conv_mask = []
        conv_templates = []
        conv_template_ids = []
        conv_transformations = []
        conv_transformation_ids = []
        conv_turn_types = []
        conv_tgt_turns = []
        turn = 0
        memory_queue = deque()
        num_dequeues = 0
        while turn < args.max_turns:
            template_id = np.random.choice(len(templates), p=p)
            template = templates[template_id]
            turn_type = "-"
            if (
                args.memory_queue
                and template_id == memory_template_id
                and len(memory_queue) >= args.max_queue_size
            ):
                # If the queue is full, don't use this template.
                continue

            # Identify transformation
            transform_lst = transformations[template_id]
            count = sum(t == template_id for t in conv_template_ids)
            transformation_id = count % len(transform_lst)
            cycle_number = count // len(transform_lst)
            transformation = transform_lst[transformation_id]
            if turn == 0:
                turn_type = "first_turn"
            elif transformation_id == 0 and (
                cycle_number == 0 or len(transform_lst) == 1
            ):
                turn_type = "multi_turn_single_template"
            elif cycle_number == 0:
                turn_type = "multi_turn_multi_template"
            else:
                turn_type = "multi_turn_cycling"

            # Sample a sentence...
            _, sentence = sample_sentence_for_template(
                template,
                min_copy_len=args.min_copy_len,
                max_copy_len=args.max_copy_len,
                unigram_concentration=args.unigram_concentration,
            )
            # ...and make sure it doesn't match a lower-ranked template.
            first_match, _ = find_matches(templates, sentence)
            while first_match < template_id:
                _, sentence = sample_sentence_for_template(
                    template,
                    min_copy_len=args.min_copy_len,
                    max_copy_len=args.max_copy_len,
                    unigram_concentration=args.unigram_concentration,
                )
                first_match, _ = find_matches(templates, sentence)

            # Memory: enqueue
            if args.memory_queue and template_id == memory_template_id:
                memory_queue.append((turn, sentence))

            # Memory: maybe dequeue
            tgt_turn = turn
            if args.memory_queue and is_null_template(template) and len(memory_queue):
                turn_type = "dequeue"
                tgt_turn, tgt_sentence = memory_queue.popleft()
                template = templates[memory_template_id]
                transformation = memory_transformations[
                    num_dequeues % len(memory_transformations)
                ]
                _, response = do_transformation(template, transformation, tgt_sentence)
                num_dequeues += 1
            elif is_null_template(template):
                turn_type = "none"
                if args.null_template_cycle_version < 2:
                    # Original version--cycle number is also incremented by dequeues.
                    _, response = do_transformation(template, transformation, sentence)
                else:
                    # v1: Don't increment cycle number for dequeues
                    count = count - num_dequeues
                    transformation_id = count % len(transform_lst)
                    cycle_number = count // len(transform_lst)
                    transformation = transform_lst[transformation_id]
                    _, response = do_transformation(template, transformation, sentence)
            else:
                _, response = do_transformation(template, transformation, sentence)

            words = list(map(str, sentence + response))
            if len(conv_words) + len(words) > args.max_conversation_len:
                break

            if turn == 0 and not args.no_bos:
                words = ["$"] + words

            r_idx = words.index("E")
            mask = [int(i > r_idx) for i in range(len(words))]
            conv_words += words
            conv_mask += mask
            conv_templates.append(" ".join(template))
            conv_template_ids.append(template_id)
            conv_transformations.append(" ".join(map(str, transformation)))
            conv_transformation_ids.append(transformation_id)
            conv_turn_types.append(turn_type)
            conv_tgt_turns.append(tgt_turn)
            turn += 1

        rows.append(
            {
                "conv_id": conv_id,
                "input": " ".join(conv_words),
                "mask": " ".join(map(str, conv_mask)),
                "templates": ";".join(conv_templates),
                "template_ids": ";".join(map(str, conv_template_ids)),
                "transformations": ";".join(conv_transformations),
                "transformation_ids": ";".join(map(str, conv_transformation_ids)),
                "turn_types": ";".join(conv_turn_types),
                "tgt_turns": ";".join(map(str, conv_tgt_turns)),
            }
        )
    return pd.DataFrame(script), pd.DataFrame(rows)


def generate_and_write_data(args):
    script, rows = generate_data(args)
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f)
    script_fn = output_dir / "script.csv"
    print(f"Writing {len(script)} rules to {script_fn}")
    with open(script_fn, "w") as f:
        script.to_csv(script_fn, index=False)
    lens = [s.count(" ") + 1 for s in rows["input"]]
    print(f"Input length: max={np.max(lens)}, mean={np.mean(lens)}")
    idxs = np.random.choice(len(rows), size=len(rows), replace=False)
    num_test = min(len(idxs) // 4, 20000)
    num_train = len(idxs) - (2 * num_test)
    train_idxs = idxs[:num_train]
    val_idxs = idxs[num_train : num_train + num_test]
    test_idxs = idxs[num_train + num_test :]
    for split, idxs in (
        ("train", train_idxs),
        ("validation", val_idxs),
        ("test", test_idxs),
    ):
        fn = output_dir / f"{split}.csv"
        print(f"Writing {len(idxs)} {split} idxs to {fn}")
        df = rows.iloc[idxs]
        df.to_csv(fn, index=False)


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    generate_and_write_data(args)
