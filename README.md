# Representing Rule-based Chatbots with Transformers

This repository contains the code for our paper, Representing Rule-based Chatbots with Transformers.
The code can be used to generate synthetic ELIZA training data, train and evaluate Transformers on ELIZA transcripts, and conduct some analysis of the learned mechanisms.
Please see our paper for more details.

## Quick links
* [Setup](#Setup)
* [Generating data](#Generating-data)
* [Training](#Training)
* [Analysis](#Analysis)
  * [Error analysis](#Error-analysis)
  * [Probing copying mechanisms](#Probing-copying-mechanisms)
  * [Generating counterfactuals](#Generating-counterfactuals)
* [Questions?](#Questions)
* [Citation](#Citation)

## Setup
Install [PyTorch](https://pytorch.org/get-started/locally/) and then install the remaining requirements: `pip install -r requirements.txt`.
This code was tested using Python 3.12 and PyTorch version 2.2.2.

## Generating data

The code we used to generate the data is in [src/generate_data.py](src/generate_data.py).
The [scripts](scripts/) directory contains the configurations for the datasets we used in our paper.
- Multi-turn conversations: [scripts/generate_multi_turn_data.sh](scripts/generate_multi_turn_data.sh)
- Single-turn conversations, varying the amount of repetition in the copying segments: [scripts/generate_single_turn_data.sh](scripts/generate_single_turn_data.sh)

The datasets we used in our experiments can also be downloaded directly from HuggingFace via this link: https://huggingface.co/datasets/danf0/eliza.

## Training models

To train Transformers on ELIZA conversations, see [src/run.py](src/run.py).
The [scripts](scripts/) directory contains the configurations for the datasets we used in our paper, for [multi-turn conversations](scripts/train_multi_turn.sh) and [single-turn conversations](scripts/train_single_turn.sh).
We ran our experiments on NVIDIA GeForce RTX 2080 Ti GPUs with 11GB of memory.
The single-turn models were trained for approximately 12 hours each, and the multi-turn models were trained for approximately 48 hours.

## Analysis

### Error analysis

The training code saves predictions on the validation set throughout training.
[src/utils/analysis_utils.py](src/utils/analysis_utils.py) contains a number of utilities for analyzing these predictions.
These can be used to measure performance as a function of various properties of the prediction, such as the turn type, the template, and the state of the memory queue.

### Probing copying mechanisms

[src/probing.py](src/probing.py) contains code examinging attention embeddings, to investigate whether attentions scores are influenced more by content or position.
To run this analysis with the configuration used in the paper, see [scripts/run_attention_probe.sh](scripts/run_attention_probe.sh).

### Counterfactual data

We generated counterfactual datasets to examine the memory mechanisms learned by the models we trained.
The code for generating these datasets is in [src/counterfactuals.py](src/counterfactuals.py).
The datasets we used in our experiments can also be downloaded directly from HuggingFace (see https://huggingface.co/datasets/danf0/eliza).
After generating a counterfactual dataset, a model can be evaluated using [src/run.py](src/run.py), using the `--eval_only` flag.

## Questions?

If you have any questions about the code or paper, please email Dan (dfriedman@cs.princeton.edu) or open an issue.

## Citation
```bibtex
@article{friedman2024representing,
  title={Representing Rule-based Chatbots with Transformers},
  author={Friedman, Dan and Panigrahi, Abhishek and Chen, Danqi},
  year={2024}
}
```
