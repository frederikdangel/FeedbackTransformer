"""
TRAINING SETUP FOR VANILLA TRANSFORMER
"""
import time

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from COGS_Benchmark import gen_src_lines
from cogs_dataset import COGSDataModule
from feedback_transformer_model import FeedbackTransformerModel
from seq2seq_model import Seq2SeqModel

trainer_flags = {
    "default_root_dir": "./cogs_benchmark_vanilla",
    "amp_backend": "native",
    "benchmark": False,
    "deterministic": False,
    "callbacks": [
        ModelCheckpoint(monitor="epoch_val_accuracy", dirpath="./cogs_benchmark_vanilla"),
        EarlyStopping(monitor="epoch_val_accuracy", mode="max", patience=2),
    ],
    "gpus": 1,
    "log_every_n_steps": 10,
    "logger": TensorBoardLogger(save_dir="logs/", name="cogs_benchmark_vanilla"),
    "max_epochs": 100,
    "progress_bar_refresh_rate": 20,
    "profiler": "simple"
}

"""
TRAIN SEQ2SEQ VANILLA TRANSFORMER FOR COGS BENCHMARK
"""

vanilla_model = Seq2SeqModel(
    model=FeedbackTransformerModel(
        encoder_feedback=False,           # disabled encoder feedback
        decoder_feedback=False,           # disabled decoder feedback
        memory_context=4,
        input_vocab_size=874,
        output_vocab_size=874,
        d_model=4,
        nhead=4,
        num_layers=3,
        dim_feedforward=8,
        max_seq_length=1000,
        dropout=0.1,
        PAD_IDX=0,
        activation="gelu",
    )
)


# INSTANTITATE A DATAMODULE
datamodule = COGSDataModule(batch_size=128, num_workers=2, use_100=False, use_Gen=True)

vanilla_trainer = pl.Trainer(**trainer_flags)
vanilla_trainer.fit(model=vanilla_model, datamodule=datamodule)

"""
TEST SEQ2SEQ VANILLA TRANSFORMER FOR COGS BENCHMARK GENERALIZATION TEST SET
"""
torch.cuda.empty_cache()

start_time = time.time()
vanilla_trainer.test()
print("--- %s seconds per sample ---" % (4*(time.time() - start_time)/len(gen_src_lines))) # test batch size is 4