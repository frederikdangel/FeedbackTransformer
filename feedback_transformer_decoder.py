import os
import math
import torch
import torch.nn.functional as F
from torch import nn

from feedback_transformer_utilities import LinearWeightedAvg, FeedforwardBlock

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


"""
FEEDBACK TRANSFORMER DECODER
"""


class FeedbackTransformerPointwiseDecoder(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_layers=2,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
    ):
        super(FeedbackTransformerPointwiseDecoder, self).__init__()

        # cross-attention encoder-decoder
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_dropout = nn.Dropout(dropout)
        self.cross_norm = nn.LayerNorm(d_model)

        # memory-attention
        self.memory_layer_wise_weighting = LinearWeightedAvg(n_inputs=num_layers)
        self.mem_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.mem_attn_dropout = nn.Dropout(dropout)
        self.mem_attn_norm = nn.LayerNorm(d_model)

        # feedforward
        self.feedforward_layers = self._get_clones(
            FeedforwardBlock(d_model, dim_feedforward, dropout, activation), num_layers
        )

    def forward(self, tgt_embed, memory_states, encoder_outputs):
        # layer-wise memory-attention and feedforward
        layer_wise_outputs = []
        memory_states = torch.stack(memory_states)
        output_embed = tgt_embed

        for feedforward in self.feedforward_layers:

            # memory-attention
            mem_attn_out, _ = self.mem_attn(
                query=output_embed, key=memory_states, value=memory_states
            )
            output_embed = output_embed + self.mem_attn_dropout(mem_attn_out)
            output_embed = self.mem_attn_norm(output_embed)

            # cross-attention to encoder outputs
            output_embed2, _ = self.cross_attn(
                output_embed, encoder_outputs, encoder_outputs
            )
            output_embed = output_embed + self.cross_dropout(output_embed2)
            output_embed = self.cross_norm(output_embed)

            # feedforward
            output_embed = feedforward(output_embed)
            layer_wise_outputs.append(output_embed)

        # output memory-state for current time-step
        output_memory_state = self.memory_layer_wise_weighting(layer_wise_outputs)
        return output_embed, output_memory_state

    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class FeedbackTransformerDecoder(nn.Module):
    def __init__(
        self,
        memory_context=16,
        d_model=256,
        nhead=8,
        num_layers=2,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
    ):
        super(FeedbackTransformerDecoder, self).__init__()

        self.memory_context = memory_context
        self.d_model = d_model

        self.pointwise_decoder = FeedbackTransformerPointwiseDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt,
        encoder_outputs,
        tgt_mask,
        memory_mask,
        tgt_key_padding_mask,
        memory_key_padding_mask,):
        # memory context
        bs = tgt.shape[1]
        memory_states = [torch.zeros(bs, self.d_model).to(tgt.device)]

        # iterate over entire sequence-length
        pred_seq_logits = []
        for i in range(tgt.shape[0]):
            output_embed, output_memory_state = self.pointwise_decoder(
                torch.unsqueeze(tgt[i], dim=0), memory_states, encoder_outputs
            )
            pred_seq_logits.append(torch.squeeze(output_embed))

            # limit the memory context
            if len(memory_states) < self.memory_context:
                memory_states = [torch.squeeze(output_memory_state)] + memory_states
            else:
                memory_states = [torch.squeeze(output_memory_state)] + memory_states[:-1]

        pred_seq_logits = self.norm(torch.stack(pred_seq_logits))
        return pred_seq_logits

