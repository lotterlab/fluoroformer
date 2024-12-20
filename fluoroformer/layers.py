import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import einops


class NLLSurvLoss(nn.Module):
    """
    Calculates the negative log likelihood loss for survival prediction with
    censoring.

    Attributes
    ----------
    eps :  float, optional
        A small value to prevent numerical instability. Defaults to 1e-8.
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, logits, label, censor):
        """
         Computes the loss for a batch of data.

        Parameters
        ----------
        logits : torch.Tensor, shape (batch_size, num_time_steps)
            Predicted logits from the model
        label :  torch.Tensor, shape (batch_size,)
            Observed event times (or censoring times)
        censor : torch.Tensor, shape (batch_size,)
            Censoring indicators--1 for censored events (i.e., alive) and
            0 for uncensored (i.e., deceased).

         Returns
         -------
             torch.Tensor, shape (1,)
             The mean negative log likelihood loss.
        """
        # Compute hazard, survival, and offset survival
        haz = logits.sigmoid() + self.eps  # prevent log(0) downstream
        sur = torch.cumprod(1 - haz, dim=1)
        sur_pad = torch.cat([torch.ones_like(censor), sur], dim=1)

        # Get values at ground truth bin
        sur_pre = sur_pad.gather(dim=1, index=label)
        sur_cur = sur_pad.gather(dim=1, index=label + 1)
        haz_cur = haz.gather(dim=1, index=label)

        # Compute NLL loss
        loss = (
            -(1 - censor) * sur_pre.log()
            - (1 - censor) * haz_cur.log()
            - censor * sur_cur.log()
        )

        return loss.mean()


class PatchAttention(nn.Module):
    """
    Implements attention pooling for patch embeddings, following the Attention-Based
    Multiple Instance Learning (ABMIL) mechanism.

    Parameters
    ----------
    input_dim : int
        The dimensionality of the input feature vectors.
    hidden_dim : int
        The dimensionality of the hidden attention vectors (u and v).
    dropout : float, optional
        Dropout probability applied after the linear transformation. Default is 0.1.

    Methods
    -------
    forward(x)
        Applies the attention mechanism and pools the patch embeddings.

    Notes
    -----
    - Attention weights are computed as the product of the hyperbolic tangent of v
      and the sigmoid of u, passed through a softmax function.
    - The final feature vector is computed as a weighted sum of the input embeddings
      using the attention weights.
    """
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.lin_uv = nn.Linear(input_dim, 2 * hidden_dim)
        self.lin_w = nn.Linear(hidden_dim, 1)
        # Assume input shape is b p e, so pool along p
        self.softmax = nn.Softmax(1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Linear transform and dropout on both UV
        uv = self.lin_uv(x)
        uv = self.dropout(uv)
        u, v = uv.chunk(2, dim=-1)

        # Attention weights via ABMIL
        w = self.lin_w(v.tanh() * u.sigmoid())
        attn = self.softmax(w)

        # Pool along patch dimension
        x = einops.reduce(x * attn, "b p e -> b e", "sum")

        return x, attn


class SDPA(nn.Module):
    """
    Implements Scaled Dot-Product Attention (SDPA) for self-attention mechanisms.

    Parameters
    ----------
    hidden_dim : int
        The dimensionality of the input and output feature vectors.

    Methods
    -------
    forward(x)
        Computes the attention scores and applies them to the value vectors.

    Notes
    -----
    - This implementation includes the computation of query, key, and value vectors
      through a single linear transformation.
    - Attention scores are scaled by the square root of the hidden dimension to
      stabilize gradients.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.scale = math.sqrt(hidden_dim)
        # Attention matrix is of shape (b p) m m, so pool along last dimension
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query, key, value = self.qkv(x).chunk(3, dim=-1)
        key_t = einops.rearrange(key, "bp m h -> bp h m")
        prod = query @ key_t
        attn = self.softmax(prod / self.scale)
        opt = attn @ value
        return opt, attn


class MarkerAttention(nn.Module):
    """
    Implements marker-level attention to fuse information across multiplexed marker
    channels for multi-channel image data.

    This module performs co-attention between marker channels, processes the embeddings
    through a bottleneck, applies skip connections, and normalizes the outputs. The
    final output is a fused embedding for each patch with pooled information from the
    marker dimension.

    Parameters
    ----------
    embedding_dim : int
        The dimensionality of the input embeddings for each marker.
    hidden_dim : int
        The dimensionality of the intermediate embeddings after the bottleneck.
    num_heads : int, optional
        Number of attention heads. Default is 1.
    dropout : float, optional
        Dropout probability applied during the forward pass. Default is 0.1.

    Methods
    -------
    forward(x)
        Computes fused patch embeddings by applying marker-level co-attention.

    Notes
    -----
    - This module operates on input tensors where the marker dimension is explicitly
      represented, making it suitable for multiplexed imaging data.
    - Skip connections and layer normalization ensure stable training and preserve
      information from the input features.
    """
    def __init__(self, embedding_dim, hidden_dim, num_heads=1, dropout=0.1):
        super(MarkerAttention, self).__init__()
        self.lin_down = nn.Linear(embedding_dim, hidden_dim)
        self.sdpa = SDPA(hidden_dim)
        self.lin_up = nn.Linear(hidden_dim, embedding_dim)
        self.gelu = nn.GELU()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        b, m, e, p = x.shape

        # Reshape to flatten along marker dimension
        x = einops.rearrange(x, "b m e p -> (b p) m e")
        skip2 = x

        # Bottleneck
        x = self.lin_down(x)  # (b p) m h
        x = self.gelu(x)
        skip1 = x

        # Co-attention
        x, attn = self.sdpa(x)  # (b p) m h, (b p) m m

        # First skip and normalize
        x = x + skip1
        x = self.norm1(x)

        # Undo bottleneck
        x = self.lin_up(x)
        x = self.gelu(x)

        # Second skip and normalize
        x = x + skip2
        x = self.norm2(x)

        # Reshape to input dims
        x = einops.rearrange(x, "(b p) m e -> b p m e", b=b, p=p)

        # Pool along marker dimension
        x = einops.reduce(x, "b p m e -> b p e", "mean")

        return x, attn
