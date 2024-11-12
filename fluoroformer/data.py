import os
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
import numpy as np

import pytorch_lightning as pl
from torch.nn.functional import one_hot
from pathlib import Path


class EmbeddedDataset(Dataset):
    def __init__(self, file_names, cutoffs, device):
        super().__init__()
        self.file_names = file_names
        self.cutoffs = cutoffs
        self.device = device

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        folder = Path(self.file_names[idx])
        # embeddings, time_, censor = torch.load(folder.with_suffix(".pt"), map_location=self.device)
        embeddings, time_, deceased_ = torch.load(folder / "emb.pt", map_location=self.device)
        time_bin = (torch.bucketize(time_, boundaries=self.cutoffs)
                .unsqueeze(0)
        )
        censor = (~torch.tensor([deceased_])).float()
        return embeddings, time_bin, time_, censor


class EmbeddedDataModule(pl.LightningDataModule):
    def __init__(self, config_path, cutoffs, device_idx):
        super().__init__()
        self.config_path = config_path
        self.device = f"cuda:{device_idx}" if device_idx != -1 else "cpu"
        self.cutoffs = torch.tensor(cutoffs).to(self.device)

    def setup(self, stage=None):
        with open(self.config_path, "r") as file:
            config = yaml.safe_load(file)
        self.train_dataset = EmbeddedDataset(
            config["train"],
            cutoffs=self.cutoffs,
            device=self.device,
        )
        self.val_dataset = EmbeddedDataset(
            config["val"], cutoffs=self.cutoffs, device=self.device,
        )
        self.test_dataset = EmbeddedDataset(
            config["test"], cutoffs=self.cutoffs, device=self.device,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1)
