from typing import *
from math import ceil

import numpy as np
from tqdm import tqdm
from skimage.transform import downscale_local_mean
from skimage import filters, color
import einops
import torch

from .utils import *


def segment(thumb: np.array):
    """
    Segments the foreground of an H&E thumbnail image using Otsu's thresholding.

    Parameters
    ----------
    thumb : np.array
        The input thumbnail image as a NumPy array of shape (H, W, C), where C is the
        number of color channels.

    Returns
    -------
    mask : np.array
        A binary mask of shape (H, W), where `True` indicates foreground and `False`
        indicates background.
    """
    im_gray = color.rgb2gray(thumb)
    thres = filters.threshold_otsu(im_gray)
    mask = im_gray < thres
    return mask


def embed(
    patches: np.array,
    model: torch.nn.Module,
    transform: torch.nn.Module,
    device: torch.device,
    batch_size: int = 64,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Computes embeddings for patches from an H&E image using a deep learning model.

    Parameters
    ----------
    patches : np.array
        A NumPy array of shape (N, H, W, C), where N is the number of patches,
        H and W are the height and width of each patch, and C is the number of channels.
    model : torch.nn.Module
        A PyTorch model used to compute the embeddings.
    transform : torch.nn.Module
        A preprocessing transformation to apply to the image patches before passing
        them to the model.
    device : torch.device
        The device (CPU or GPU) on which to perform the computations.
    batch_size : int, optional
        The number of patches to process in each batch. Default is 64.
    verbose : bool, optional
        Whether to display a progress bar during embedding. Default is True.

    Returns
    -------
    opt_embs : torch.Tensor
        A tensor of shape (N, E), where E is the embedding size produced by the model
        for each patch.
    """
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
    opt_embs = torch.cat(opt_embs, dim=0)  # b e

    return opt_embs


def process_hne(
    im: np.ndarray,
    model: torch.nn.Module,
    transform: torch.nn.Module,
    device: torch.device,
    batch_size: int = 64,
    patch_size: int = 224,
    verbose: bool = True,
) -> Tuple[torch.Tensor, np.array, np.array]:
    """
    Processes an H&E image by cropping, downsampling, segmenting, and embedding
    image patches using a deep learning model.

    Parameters
    ----------
    im : np.ndarray
        The input H&E image as a NumPy array of shape (H, W, C), where H is the height,
        W is the width, and C is the number of color channels.
    model : torch.nn.Module
        A PyTorch model used to compute the embeddings of the image patches.
    transform : torch.nn.Module
        A preprocessing transformation to apply to the image patches before passing
        them to the model.
    device : torch.device
        The device (CPU or GPU) on which to perform the computations.
    batch_size : int, optional
        The number of patches to process in each batch. Default is 64.
    patch_size : int, optional
        The size of the square patches (in pixels) used during patch extraction.
        Default is 224.
    verbose : bool, optional
        Whether to display progress during the embedding process. Default is True.

    Returns
    -------
    thumb : np.array
        A downsampled thumbnail of the cropped image with shape
        (H', W', C), where H' and W' are the downscaled dimensions, and C is the
        number of color channels.
    mask : np.array
        A binary segmentation mask of shape (H', W'), where `True` indicates
        foreground pixels and `False` indicates background pixels.
    emb : torch.Tensor
        A tensor of shape (N, E), where N is the number of foreground patches, and
        E is the embedding size produced by the model.

    Notes
    -----
    - The function ensures that the image dimensions are evenly divisible by `patch_size`
      by cropping the image.
    - The thumbnail is created by downsampling the cropped image using local mean pooling.
    - Foreground segmentation is performed using Otsu's thresholding on the grayscale
      version of the thumbnail.
    - Patches marked as foreground by the segmentation mask are extracted and embedded
      using the provided model and transformation pipeline.
    - Assertions verify that the number of embeddings matches the foreground patches
      identified by the mask and that the mask dimensions match the thumbnail's spatial dimensions.
    """
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
