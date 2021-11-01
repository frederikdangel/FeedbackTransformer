"""
FEEDBACK TRANSFORMER UTILITIES
"""
import os
import math
import torch
import torch.nn.functional as F
from torch import nn
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import random
import string
import pandas as pd
import copy
import time
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=400):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class LinearWeightedAvg(nn.Module):
    def __init__(self, n_inputs):
        super(LinearWeightedAvg, self).__init__()
        self.weights = nn.Parameter(torch.randn(n_inputs))
        self.softmax = nn.Softmax(dim=0)

        def forward(self, input):
            res = 0
            weights = self.softmax(self.weights)
            for emb_idx, emb in enumerate(input):
                res += emb * weights[emb_idx]
            return res


class FeedforwardBlock(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1, activation="gelu"):
        super(FeedforwardBlock, self).__init__()

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout_projection = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout_residual = nn.Dropout(dropout)
        self.norm_ff = nn.LayerNorm(d_model)
        self.dropout_ff = nn.Dropout(dropout)
        self.activation = self._get_activation_fn(activation)

    def forward(self, x):
        hidden_state = self.dropout_projection(
            self.activation(self.linear1(x))
        )  # projection
        ff_output = self.dropout_ff(self.linear2(hidden_state))  # feed-forward
        output = x + self.dropout_residual(ff_output)  # residual-connection
        output = self.norm_ff(output)
        return output

    def _get_activation_fn(self, activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
