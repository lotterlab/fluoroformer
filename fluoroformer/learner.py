from pathlib import Path
from typing import *

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import *
from pytorch_lightning.loggers import WandbLogger
from torch import optim
import einops

import numpy as np
from lifelines.utils import concordance_index

from fluoroformer.layers import *


class Learner(pl.LightningModule):
    def __init__(
        self,
        mif: bool,
        embed_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_cutoffs: int,
        dropout: float,
        test_dir: Path = Path("."),
    ) -> None:
        super(Learner, self).__init__()

        self.save_hyperparameters()

        if self.hparams.mif:
            self.marker_attention = MarkerAttention(
                embedding_dim=self.hparams.embed_dim,
                hidden_dim=self.hparams.hidden_dim,
                num_heads=self.hparams.num_heads,
                dropout=self.hparams.dropout,
            )
            self.patch_attention = PatchAttention(
                input_dim=self.hparams.embed_dim,
                hidden_dim=self.hparams.hidden_dim,
                dropout=self.hparams.dropout,
            )
            self.linear = nn.Linear(
                self.hparams.embed_dim, self.hparams.num_cutoffs + 1
            )
        else:
            self.patch_attention = PatchAttention(
                input_dim=self.hparams.embed_dim,
                hidden_dim=self.hparams.hidden_dim,
                dropout=self.hparams.dropout,
            )
            self.linear = nn.Linear(
                self.hparams.embed_dim, self.hparams.num_cutoffs + 1
            )

        self.criterion = NLLSurvLoss()

        self.train_data_list = []
        self.val_data_list = []
        self.test_data_list = []

    def forward(self, x, return_attention=False):
        if self.hparams.mif:
            x = einops.rearrange(x, "b p m e -> b m e p")
            x, marker_weights = self.marker_attention(x)
        else:
            marker_weights = None
        x, patch_weights = self.patch_attention(x)
        logits = self.linear(x)
        if return_attention:
            return logits, (marker_weights, patch_weights)
        return logits

    def _shared_step(self, batch: tuple, stage: str, batch_idx: int) -> torch.Tensor:
        # Perform inference
        inputs, grd, grd_cont, censor = batch
        logits, (marker_weights, patch_weights) = self(inputs, return_attention=True)

        # Compute and log loss
        loss = self.criterion(logits, grd, censor)
        self.log(f"{stage}/loss", loss, prog_bar=True)

        # Save outputs if test phase
        if stage == "test":
            torch.save(
                (marker_weights, patch_weights, logits, grd_cont, censor),
                self.hparams.test_dir / f"prd_{batch_idx:03}.pt",
            )

        # Cache outputs for c-index computation
        opt_row = logits.sigmoid().squeeze().tolist() + [grd_cont.detach(), censor]
        getattr(self, f"{stage}_data_list").append(opt_row)

        return loss

    def on_fit_start(self):
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log_code("train")

    def on_test_start(self):
        self.hparams.test_dir.mkdir(exist_ok=True)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train", batch_idx)

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        self._shared_step(batch, "val", batch_idx)

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        self._shared_step(batch, "test", batch_idx)

    def _shared_epoch_end(self, stage: str):
        data_list = getattr(self, f"{stage}_data_list")

        if not self.trainer.is_global_zero or not data_list:
            return

        data_torch = self.all_gather(torch.tensor(data_list))

        # Reject invalid rows
        valid_mask = ~data_torch[:, -1].bool()
        nan_mask = ~data_torch.sum(1).isnan()
        data_torch = data_torch[nan_mask]

        # Chunk dataframe into constituent columns
        censor = data_torch[:, -1]
        grd_cont = data_torch[:, -2]
        hazards = data_torch[:, :-2]
        hazards_bin = hazards.chunk(self.hparams.num_cutoffs + 1, dim=1)

        # Compute risk score
        surv = torch.cumprod(1 - hazards, dim=1)
        # risk = -torch.sum(surv, dim=1)
        neg_risk = torch.sum(surv, dim=1)

        # C-indices for each hazard bin
        c_indices = []
        for i, bin_ in enumerate(hazards_bin):
            c_index = self._get_cindex(grd_cont, bin_, censor)
            c_indices.append(c_index)
            self.log(f"{stage}/cindex_{i}", c_index)
        self.log(f"{stage}/cindex_max", max(c_indices))

        # Global c-index for risk score
        c_index_risk = self._get_cindex(grd_cont, neg_risk, censor)
        self.log(f"{stage}/cindex_risk", c_index_risk)

        # Reset dataframe
        setattr(self, f"{stage}_data_list", [])

    def on_validation_epoch_end(self):
        self._shared_epoch_end("val")

    def on_test_epoch_end(self):
        self._shared_epoch_end("test")

    def on_train_epoch_end(self):
        self._shared_epoch_end("train")

    def _get_cindex(self, grd, prd, censor):
        try:
            # N.B.: Lifelines uses event, not censor!
            event = 1 - censor
            c_index = concordance_index(
                grd.cpu().numpy(), prd.cpu().numpy(), event.cpu().numpy()
            )
        except:
            c_index = -np.inf
        return c_index
