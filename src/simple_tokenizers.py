from pathlib import Path

import numpy as np
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.models import WordLevel

PAD = "<pad>"
UNK = "<unk>"


def make_tokenizer(path):
    fn = Path(path) / "train.csv"
    df = pd.read_csv(fn)
    sents = df["input"]
    words = [PAD] + sorted(set(w for s in sents for w in s.split(" ")))
    vocab = {w: i for i, w in enumerate(words)}
    vocab[UNK] = len(vocab)
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token=UNK))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    fn_out = Path(path) / "tokenizer.json"
    print(f"Saving tokenizer with {len(vocab)} words to {fn_out}")
    tokenizer.save(str(fn_out))
    return tokenizer


def get_tokenizer(path, use_cache=True):
    fn = Path(path) / "tokenizer.json"
    if not fn.exists() or not use_cache:
        print(f"Generating tokenizer for {path}")
        tokenizer = make_tokenizer(path)
    else:
        tokenizer = Tokenizer.from_file(str(fn))
    idx_w = np.array(
        [tokenizer.id_to_token(i) for i in range(tokenizer.get_vocab_size())]
    )
    return tokenizer, idx_w
