from typing import *
import os
from math import ceil
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

import tifffile
from skimage.transform import downscale_local_mean
from skimage import filters

import xmltodict
import xml.etree.ElementTree as ET

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login

import einops
import torch
from torchvision import transforms
from torchvision.transforms.v2 import *
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn

MARKERS = ['DAPI', 'FOXP3', 'Cytokeratin', 'CD8', 'PD-1', 'PD-L1', 'Autofluorescence', 'ROI']

class HeadlessResNet50(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

    def forward(self, x):
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        return x

def crop(im: np.array, patch_size: int):
    # compute number of patches
    height, width, _ = im.shape
    n_patches_h = height // patch_size
    n_patches_w = width // patch_size

    # crop to be evenly divisible by patch_size
    height_crop = patch_size * n_patches_h
    width_crop = patch_size * n_patches_w
    im = im[:height_crop, :width_crop, :]

    return im, n_patches_h, n_patches_w


def segment(thumb: np.array):
    # masks for all markers
    fore_mask = 0
    hard_mask = 0

    # chunk along marker dimension
    ims = np.split(thumb, thumb.shape[-1], axis=-1)

    for im in ims:
        # get hard background and foreground
        hard_mask_cur = im == 0
        fore_mask_cur = im > filters.threshold_otsu(im[~hard_mask_cur])

        # update global masks
        fore_mask += fore_mask_cur
        hard_mask += hard_mask_cur

    # pixel is foreground if foreground in any one marker
    fore_mask = fore_mask > 0

    # pixel is hard background if hard background in any one marker
    fore_mask[hard_mask > 0] = False

    return fore_mask


def patchify(
    im: np.array, mask: np.array, patch_size: int, n_patches_h: int, n_patches_w: int
):
    patches = []
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            # get image indices
            start_i = i * patch_size
            end_i = start_i + patch_size
            start_j = j * patch_size
            end_j = start_j + patch_size

            # get patch
            patch = im[start_i:end_i, start_j:end_j, :]

            # append if in foreground
            if mask[i, j]:
                patches.append(patch)

    patches = np.stack(patches)

    return patches


def embed(
    patches: np.array,
    model: torch.nn.Module,
    transform: torch.nn.Module,
    device: torch.device,
    batch_size: int = 64,
    verbose: bool = True,
) -> torch.Tensor:
    num_batches = ceil(len(patches) / batch_size)
    opt_embs = []

    for batch_idx in tqdm(range(num_batches), disable=not verbose):
        # slice batch
        start = batch_idx * batch_size
        end = min(start + batch_size, len(patches))
        batch_np = patches[start:end]

        # copy to device, flip channels, normal to [0, 1]
        batch = torch.from_numpy(batch_np).to(device).float() / 255.0

        # rearrange markers
        b, h, w, m = batch.shape
        batch = einops.rearrange(batch, "b h w m -> b m h w")

        # subtract autofluorescence and clip back to [0, 1]
        auto = batch[:, -1, ...].unsqueeze(1)
        batch[:, :-1, ...] -= auto
        batch = batch.clip(0, 1)

        # flatten along marker dimension
        batch = einops.rearrange(batch, "b m h w -> (b m) h w")

        # expand to be of shape (b m) c h w
        batch = batch.unsqueeze(1)
        batch = batch.repeat_interleave(3, dim=1)

        # use model transform
        batch = transform(batch)

        # call model
        with torch.no_grad():
            batch_emb = model(batch)

        # unflatten along marker dimension
        batch_emb = einops.rearrange(batch_emb, "(b m) e -> b m e", b=b, m=m)

        # copy to host and append
        opt_embs.append(batch_emb.cpu())

    # stack to contiguous array
    opt_embs = torch.cat(opt_embs, dim=0)

    return opt_embs


def process_mif(
    ipt_path: Path,
    model: torch.nn.Module,
    transform: torch.nn.Module,
    device: torch.device,
    batch_size: int = 64,
    patch_size: int = 224,
    verbose: bool = True,
) -> Tuple[torch.Tensor, np.array, np.array]:
    with tifffile.TiffFile(ipt_path) as tif:
        data = xmltodict.parse(tif.pages[0].description)
        names_wsi = [d["@Name"] for d in data["OME"]["Image"]["Pixels"]["Channel"]]
        assert names_wsi == MARKERS
        im = tif.asarray()

    # transpose to put markers last
    im = np.transpose(im, (1, 2, 0))
    im = im[..., :7]

    im_crop, n_patches_h, n_patches_w = crop(im, patch_size)

    # downsample down to patch_size
    thumb = downscale_local_mean(im_crop, (patch_size, patch_size, 1))

    # segment image
    mask = segment(thumb)

    # patchify using mask
    patches = patchify(im, mask, patch_size, n_patches_h, n_patches_w)

    # embed
    emb = embed(
        patches=patches,
        model=model,
        transform=transform,
        device=device,
        batch_size=batch_size,
        verbose=verbose,
    )

    assert emb.shape[0] == mask.sum()
    assert mask.shape[:-1] == thumb.shape[:-1]

    return thumb, mask, emb
