from torch import nn
from transformers import GPT2LMHeadModel


class NoPE(nn.Module):
    def forward(self, *args, **kwargs):
        return 0


class GPTNoPE(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer.wpe = NoPE()
