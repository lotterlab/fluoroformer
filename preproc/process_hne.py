from typing import *
import os
from math import ceil
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

import tifffile
from skimage.transform import downscale_local_mean
from skimage import filters, color

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

MARKERS = [
    "DAPI",
    "FOXP3",
    "Cytokeratin",
    "CD8",
    "PD-1",
    "PD-L1",
    "Autofluorescence",
    "ROI",
]


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
    # Compute number of patches
    height, width, _ = im.shape
    n_patches_h = height // patch_size
    n_patches_w = width // patch_size

    # Crop to be evenly divisible by patch_size
    height_crop = patch_size * n_patches_h
    width_crop = patch_size * n_patches_w
    im = im[:height_crop, :width_crop, :]

    return im, n_patches_h, n_patches_w


def segment(thumb: np.array):
    im_gray = color.rgb2gray(thumb)
    thres = filters.threshold_otsu(im_gray)
    mask = im_gray < thres
    return mask


def patchify(
    im: np.array, mask: np.array, patch_size: int, n_patches_h: int, n_patches_w: int
):
    patches = []
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            # Get image indices
            start_i = i * patch_size
            end_i = start_i + patch_size
            start_j = j * patch_size
            end_j = start_j + patch_size

            # Get patch
            patch = im[start_i:end_i, start_j:end_j, :]

            # Append if in foreground
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
        # Slice batch
        start = batch_idx * batch_size
        end = min(start + batch_size, len(patches))
        batch_np = patches[start:end]

        # Copy to device, flip channels, normal to [0, 1]
        batch = torch.from_numpy(batch_np).to(device).float() / 255.0

        # Rearrange markers
        b, h, w, c = batch.shape
        batch = einops.rearrange(batch, "b h w c -> b c h w")

        # Use model transform
        batch = transform(batch)

        # Call model
        with torch.no_grad():
            batch_emb = model(batch)

        # Copy to host and append
        opt_embs.append(batch_emb.cpu())

    # Stack to contiguous array
    opt_embs = torch.cat(opt_embs, dim=0) # b e

    return opt_embs


def process_hne(
    ipt_path: Path,
    model: torch.nn.Module,
    transform: torch.nn.Module,
    device: torch.device,
    batch_size: int = 64,
    patch_size: int = 224,
    verbose: bool = True,
) -> Tuple[torch.Tensor, np.array, np.array]:

    with tifffile.TiffFile(ipt_path) as tif:
        im = tif.asarray()

    # Crop to make evenly divisible by 224
    im_crop, n_patches_h, n_patches_w = crop(im, patch_size)

    # Downsample down to patch_size
    thumb = downscale_local_mean(im_crop, (patch_size, patch_size, 1))

    # Segment image
    mask = segment(thumb)

    # Patchify using mask
    patches = patchify(im, mask, patch_size, n_patches_h, n_patches_w)

    # Embed
    emb = embed(
        patches=patches,
        model=model,
        transform=transform,
        device=device,
        batch_size=batch_size,
        verbose=verbose,
    )

    assert emb.shape[0] == mask.sum()
    assert mask.shape == thumb.shape[:-1]

    return thumb, mask, emb
