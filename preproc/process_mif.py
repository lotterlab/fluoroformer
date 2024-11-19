from typing import *
from math import ceil

import numpy as np
from tqdm import tqdm
from skimage.transform import downscale_local_mean
from skimage import filters
import einops
import torch

from .utils import *


def segment(thumb: np.array):
    """
    Segments the foreground of a multiplexed image by combining masks for all markers.

    Parameters
    ----------
    thumb : np.array
        The input multiplexed image as a NumPy array of shape (H, W, M), where H is
        the height, W is the width, and M is the number of markers.

    Returns
    -------
    fore_mask : np.array
        A binary mask of shape (H, W), where `True` indicates foreground pixels
        and `False` indicates background pixels. Pixels identified as hard
        background in any marker are excluded from the foreground mask.

    """
    # Masks for all markers
    fore_mask = 0
    hard_mask = 0

    # Chunk along marker dimension
    ims = np.split(thumb, thumb.shape[-1], axis=-1)

    for im in ims:
        # Get hard background and foreground
        hard_mask_cur = im == 0
        fore_mask_cur = im > filters.threshold_otsu(im[~hard_mask_cur])

        # Update global masks
        fore_mask += fore_mask_cur
        hard_mask += hard_mask_cur

    # Pixel is foreground if foreground in any one marker
    fore_mask = fore_mask > 0

    # Pixel is hard background if hard background in any one marker
    fore_mask[hard_mask > 0] = False

    return fore_mask


def embed(
    patches: np.array,
    model: torch.nn.Module,
    transform: torch.nn.Module,
    device: torch.device,
    batch_size: int = 64,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Embeds a batch of image patches using a deep learning model and preprocessing pipeline.

    Parameters
    ----------
    patches : np.array
        A NumPy array of shape (N, H, W, M), where N is the number of patches,
        H and W are the height and width of each patch, and M is the number of markers.
    model : torch.nn.Module
        A PyTorch model used to compute the embeddings.
    transform : torch.nn.Module
        A preprocessing transformation to apply to the image patches before
        passing them to the model.
    device : torch.device
        The device (CPU or GPU) on which to perform the computations.
    batch_size : int, optional
        The number of patches to process in each batch. Default is 64.
    verbose : bool, optional
        Whether to display a progress bar during the embedding process. Default is True.

    Returns
    -------
    opt_embs : torch.Tensor
        A tensor of shape (N, M, E), where E is the embedding size produced by the model
        for each marker in each patch.

    Notes
    -----
    - The function normalizes the input patches to a [0, 1] range by dividing by 255.0.
    - Autofluorescence is subtracted from each marker channel (except the last one).
    - The markers are flattened, expanded into a pseudo-RGB format, and transformed before
      being passed to the model.
    - The embeddings for each patch are restructured back into the original patch/marker layout.
    - The function ensures memory efficiency by processing the patches in batches and using
      PyTorch's `no_grad` context to disable gradient computation.

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
        b, h, w, m = batch.shape
        batch = einops.rearrange(batch, "b h w m -> b m h w")

        # Subtract autofluorescence and clip back to [0, 1]
        auto = batch[:, -1, ...].unsqueeze(1)
        batch[:, :-1, ...] -= auto
        batch = batch.clip(0, 1)

        # Flatten along marker dimension
        batch = einops.rearrange(batch, "b m h w -> (b m) h w")

        # Expand to be of shape (b m) c h w
        batch = batch.unsqueeze(1)
        batch = batch.repeat_interleave(3, dim=1)

        # Use model transform
        batch = transform(batch)

        # Call model
        with torch.no_grad():
            batch_emb = model(batch)

        # Unflatten along marker dimension
        batch_emb = einops.rearrange(batch_emb, "(b m) e -> b m e", b=b, m=m)

        # Copy to host and append
        opt_embs.append(batch_emb.cpu())

    # Stack to contiguous array
    opt_embs = torch.cat(opt_embs, dim=0)

    return opt_embs


def process_mif(
    im: np.array,
    model: torch.nn.Module,
    transform: torch.nn.Module,
    device: torch.device,
    batch_size: int = 64,
    patch_size: int = 224,
    verbose: bool = True,
) -> Tuple[torch.Tensor, np.array, np.array]:
    """
    Processes a multiplexed image file (MIF) by cropping, downsampling, segmenting,
    and embedding the image using a deep learning model.

    Parameters
    ----------
    im : np.array
        The input multiplexed image as a NumPy array of shape (M, H, W), where M
        is the number of markers, H is the height, and W is the width.
    model : torch.nn.Module
        A PyTorch model used to compute the embeddings of the image patches.
    transform : torch.nn.Module
        A preprocessing transformation to apply to the image patches before
        passing them to the model.
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
        (H', W', M), where H' and W' are the downscaled dimensions and M
        is the number of markers.
    mask : np.array
        A binary segmentation mask of shape (H', W'), where `True` indicates
        foreground pixels and `False` indicates background pixels.
    emb : torch.Tensor
        A tensor of shape (N, M, E), where N is the number of foreground patches,
        M is the number of markers, and E is the embedding size produced by the model.

    Notes
    -----
    - The input image is transposed to have markers as the last dimension.
    - The image is cropped to ensure its dimensions are evenly divisible by `patch_size`.
    - A downsampled thumbnail is created using local mean pooling.
    - Foreground segmentation is performed using marker-specific thresholds.
    - Only patches marked as foreground in the mask are embedded.
    - The function asserts that the number of embeddings matches the number of foreground patches.

    """
    # Transpose to put markers last
    im = np.transpose(im, (1, 2, 0))
    im = im[..., :7]

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
    assert mask.shape[:-1] == thumb.shape[:-1]

    return thumb, mask, emb
