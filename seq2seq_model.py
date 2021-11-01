"""
LIGHTNING MODULE FOR SEQ2SEQ TASKS
"""
import torch
import torchmetrics
import pytorch_lightning as pl
from torch import nn


class Seq2SeqModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss(ignore_index=0)

    def training_step(self, batch, batch_idx):
        input_seq, output_seq = batch
        pred_seq_logits = self.model(
            input_seq, output_seq[:, :-1], flatten_lm_output=True
        )
        loss = self.loss(pred_seq_logits, output_seq[:, 1:].reshape(-1))

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        input_seq, output_seq = batch
        pred_seq_logits = self.model(
            input_seq, output_seq[:, :-1], flatten_lm_output=True
        )
        loss = self.loss(pred_seq_logits, output_seq[:, 1:].reshape(-1))

        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return {
            "loss": loss,
            "pred": pred_seq_logits,
            "ground_truth": output_seq[:, 1:].reshape(-1),
        }

    def validation_epoch_end(self, validation_step_outputs):
        preds, ground_truths = [], []
        for out in validation_step_outputs:
            preds += torch.argmax(out["pred"], dim=1).tolist()
            ground_truths += out["ground_truth"].tolist()
        accuracy = torchmetrics.functional.accuracy(preds=torch.tensor(preds), target=torch.tensor(ground_truths), ignore_index=0)

        self.log(
            "epoch_val_accuracy", accuracy, on_epoch=True, prog_bar=True, logger=True
        )

    def test_step(self, batch, batch_idx):
        input_seq, output_seq = batch
        pred_seq_logits = self.model(
            input_seq, output_seq[:, :-1], flatten_lm_output=True
        )
        loss = self.loss(pred_seq_logits, output_seq[:, 1:].reshape(-1))

        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return {
            "loss": loss,
            "pred": pred_seq_logits,
            "ground_truth": output_seq[:, 1:].reshape(-1),
        }

    def test_epoch_end(self, test_step_outputs):
        preds, ground_truths = [], []
        for out in test_step_outputs:
            preds += torch.argmax(out["pred"], dim=1).tolist()
            ground_truths += out["ground_truth"].tolist()
        accuracy = torchmetrics.functional.accuracy(preds=torch.tensor(preds), target=torch.tensor(ground_truths), ignore_index=0)

        self.log(
            "epoch_test_accuracy", accuracy, on_epoch=True, prog_bar=True, logger=True
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-3, weight_decay=0.0)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode="min", factor=0.1, patience=1
        # ) # disabled to reduce script run time
        return {
            "optimizer": optimizer,
            # "lr_scheduler": scheduler,
            "monitor": "epoch_val_accuracy",
        }