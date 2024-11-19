import os
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
import numpy as np

import pytorch_lightning as pl
from torch.nn.functional import one_hot
from pathlib import Path


class EmbeddedDataset(Dataset):
    """
    A PyTorch Dataset for loading precomputed embeddings along with survival
    times and censoring information.

    Parameters
    ----------
    file_names : list of str
        A list of file paths to directories containing embedding files (`emb.pt`).
    cutoffs : list or torch.Tensor
        Cutoff values for binning survival times into discrete intervals.
    device : torch.device
        The device (CPU or GPU) to which the data should be loaded.

    Methods
    -------
    __len__()
        Returns the number of embedding files in the dataset.
    __getitem__(idx)
        Loads the embeddings, survival time, and censoring information for the
        given index.

    Notes
    -----
    - Each `emb.pt` file is expected to contain:
        - `embeddings` : torch.Tensor, the feature embeddings.
        - `time_` : torch.Tensor, the survival times.
        - `deceased_` : bool, indicating whether the subject is deceased.
    - Survival times are bucketized into bins defined by `cutoffs`, and a censor
      value is computed as `1.0` for censored and `0.0` for uncensored observations.
    """
    def __init__(self, file_names, cutoffs, device):
        super().__init__()
        self.file_names = file_names
        self.cutoffs = cutoffs
        self.device = device

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # Get subdirectory of slide
        folder = Path(self.file_names[idx])

        # Load embeddings from subdirectory
        embeddings, time_, deceased_ = torch.load(folder / "emb.pt", map_location=self.device)

        # Discretize survival time
        time_bin = (torch.bucketize(time_, boundaries=self.cutoffs)
                .unsqueeze(0)
        )

        # Convert `deceased` Bool to `censor` Bool
        censor = (~torch.tensor([deceased_])).float()

        return embeddings, time_bin, time_, censor


class EmbeddedDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for managing and loading datasets of precomputed
    embeddings, along with associated survival times and censoring information.

    This DataModule organizes the datasets for training, validation, and testing
    based on a configuration file that specifies the dataset splits.

    Parameters
    ----------
    config_path : str
        Path to the configuration file (YAML) that defines the dataset splits. The
        configuration file should contain keys `train`, `val`, and `test`, each
        mapping to lists of directories. Each directory corresponds to a slide
        subdirectory containing the data files (e.g., `emb.pt`).
    cutoffs : list or torch.Tensor
        Cutoff values for binning survival times into discrete intervals.
    device_idx : int
        The GPU device index to use. If set to `-1`, the computations will run on CPU.

    Attributes
    ----------
    config_path : str
        Path to the configuration file.
    device : str
        The device on which computations will be performed (`"cuda:X"` or `"cpu"`).
    cutoffs : torch.Tensor
        A tensor of cutoff values for binning survival times, moved to the specified device.
    train_dataset : EmbeddedDataset
        The dataset for training, created during the `setup` phase.
    val_dataset : EmbeddedDataset
        The dataset for validation, created during the `setup` phase.
    test_dataset : EmbeddedDataset
        The dataset for testing, created during the `setup` phase.

    Methods
    -------
    setup(stage=None)
        Initializes the training, validation, and testing datasets using the configuration file.
    train_dataloader()
        Returns a DataLoader for the training dataset.
    val_dataloader()
        Returns a DataLoader for the validation dataset.
    test_dataloader()
        Returns a DataLoader for the testing dataset.

    Notes
    -----
    - The configuration file at `config_path` must be a YAML file that specifies the paths
      to training, validation, and testing data directories.
    - Each directory listed in the configuration file should correspond to a slide
      subdirectory containing the necessary data files, such as `emb.pt`.
    - The DataLoader uses a batch size of 1 for all datasets, and shuffling is enabled
      for the training DataLoader.
    """
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
