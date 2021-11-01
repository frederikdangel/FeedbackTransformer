"""
PYTORCH DATASET CLASS
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from COGS_Benchmark import train_src_lines, train_100_src_lines, train_100_tgt_lines, token2id, train_tgt_lines, \
    valid_src_lines, gen_src_lines, valid_tgt_lines, gen_tgt_lines, test_src_lines, test_tgt_lines


class COGSDataset(Dataset):
    def __init__(self, src_lines, tgt_lines, token2id):
        super().__init__()

        self.PAD_IDX = 0
        self.BOS_IDX = 1
        self.EOS_IDX = 2

        # tokenize data
        self.PAD_IDX = 0
        self.src_lines_tokenized = self.tokenize(src_lines, token2id)
        self.tgt_lines_tokenized = self.tokenize(tgt_lines, token2id)

    def tokenize(self, lines, token2id):
        # tokenize lines
        tokenized_lines = []
        for line in lines:
            tokenized_line = (
                [self.BOS_IDX]
                + [token2id[item] for item in line.split()]
                + [self.EOS_IDX]
            )
            tokenized_lines.append(tokenized_line)
        return tokenized_lines

    def __len__(self):
        return len(self.src_lines_tokenized)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        input_seq = torch.tensor(self.src_lines_tokenized[idx])
        output_seq = torch.tensor(self.tgt_lines_tokenized[idx])
        return input_seq, output_seq

    def pad_tensor(self, tensor, max_length):
        tensor = torch.cat(
            [tensor, torch.zeros(max_length - tensor.shape[0], dtype=torch.int32)],
            dim=0,
        )
        return tensor

    def collate_fn(self, batch):
        # find longest sequences
        max_len_input = max([sample[0].shape[0] for sample in batch])
        max_len_output = max([sample[1].shape[0] for sample in batch])

        # pad according to max_length
        input_seq = [self.pad_tensor(sample[0], max_len_input) for sample in batch]
        output_seq = [self.pad_tensor(sample[1], max_len_output) for sample in batch]

        # stack all
        input_seq = torch.stack(input_seq, dim=0)
        output_seq = torch.stack(output_seq, dim=0)
        return input_seq, output_seq


"""
LIGHTNING DATAMODULE FOR COGS BENCHMARK
"""


class COGSDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=2, use_100=True, use_Gen=True):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_100 = use_100
        self.use_Gen = use_Gen

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            if self.use_100:
                self.train_dataset = COGSDataset(
                    train_100_src_lines, train_100_tgt_lines, token2id
                )
            else:
                self.train_dataset = COGSDataset(
                    train_src_lines, train_tgt_lines, token2id
                )
            self.valid_dataset = COGSDataset(valid_src_lines, valid_tgt_lines, token2id)
        if stage == "test" or stage is None:
            if self.use_Gen:
                self.test_dataset = COGSDataset(gen_src_lines, gen_tgt_lines, token2id)
            else:
                self.test_dataset = COGSDataset(test_src_lines, test_tgt_lines, token2id)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_dataset.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.valid_dataset.collate_fn,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=4,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.test_dataset.collate_fn,
            drop_last=True,
        )


