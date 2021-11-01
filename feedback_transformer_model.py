"""
END TO END FEEDBACK TRANSFORMER MODEL
"""

import os
import math
import torch
import torch.nn.functional as F
from torch import nn

from feedback_transformer_decoder import FeedbackTransformerDecoder
from feedback_transformer_encoder import FeedbackTransformerEncoder
from feedback_transformer_utilities import LinearWeightedAvg, FeedforwardBlock, PositionalEncoding

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


class FeedbackTransformerModel(nn.Module):
    def __init__(
        self,
        encoder_feedback=False,
        decoder_feedback=True,
        memory_context=16,
        input_vocab_size=11,
        output_vocab_size=11,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_length=400,
        PAD_IDX=10,
        activation="gelu",
    ):
        super(FeedbackTransformerModel, self).__init__()

        self.d_model = d_model
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.PAD_IDX = PAD_IDX

        # Embeddings
        self.pos_encoder = PositionalEncoding(
            d_model, dropout=dropout, max_len=max_seq_length
        )
        self.src_embedding = nn.Embedding(input_vocab_size, d_model, padding_idx=0, sparse=False)
        self.tgt_embedding = nn.Embedding(output_vocab_size, d_model, padding_idx=0, sparse=False)

        # Feedback Transformer
        if encoder_feedback:
            feedback_encoder = FeedbackTransformerEncoder(
                memory_context=memory_context,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
            )
        else:
            feedback_encoder = None

        if decoder_feedback:
            feedback_decoder = FeedbackTransformerDecoder(
                memory_context=memory_context,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
            )
        else:
            feedback_decoder = None

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            custom_encoder=feedback_encoder,
            custom_decoder=feedback_decoder,
            dropout=dropout,
            activation=activation,
        )

        self.lm_layer = nn.Linear(d_model, output_vocab_size)

    def forward(self, input_seq, output_seq, flatten_lm_output=False):
        # Input Sequence (N,S) -> Permuted Input Sequence (S,N)
        input_seq = input_seq.permute(1, 0)
        output_seq = output_seq.permute(1, 0)

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(
            input_seq, output_seq, self.PAD_IDX
        )

        src_embeddings = self.pos_encoder(
            self.src_embedding(input_seq) * math.sqrt(self.d_model)
        )
        tgt_embeddings = self.pos_encoder(
            self.tgt_embedding(output_seq) * math.sqrt(self.d_model)
        )

        transformer_outputs = self.transformer(
            src=src_embeddings,
            tgt=tgt_embeddings,
            src_mask=src_mask.to(src_embeddings.device),
            tgt_mask=tgt_mask.to(tgt_embeddings.device),
            src_key_padding_mask=src_padding_mask.to(src_embeddings.device),
            tgt_key_padding_mask=tgt_padding_mask.to(tgt_embeddings.device),
        )

        pred_seq_logits = self.lm_layer(transformer_outputs).permute(1, 0, 2)
        if flatten_lm_output:
            pred_seq_logits = pred_seq_logits.reshape(-1, self.output_vocab_size)
        return pred_seq_logits

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def create_mask(self, src, tgt, PAD_IDX):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)

        src_padding_mask = (src == PAD_IDX).transpose(0, 1)
        tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)

        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask