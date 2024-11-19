# Fluoroformer
[![arXiv](https://img.shields.io/badge/arXiv-2411.08975-B31B1B.svg)](https://arxiv.org/abs/2411.08975)
Official implementation of the Fluoroformer module from **[Fluoroformer: Scaling multiple instance learning to multiplexed images via attention-based channel fusion](https://arxiv.org/abs/2411.08975)**.

# Core modules and training scripts

The Fluoroformer module itself, along with the basic multiple instance learning (MIL) modules, are implemented in PyTorch and located in `fluoroformer/layers.py`, while the training loop is performed by the [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) and located in `fluoroformer/learner.py`: 
Executing the training and evaluation loops is performed via the [Lightning Command-Line Interface (CLI)](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_intermediate.html) as follows:
```bash
python -m fluoroformer --help
```
See the documentation and the output of the `help` message for more details.

# Preprocessing

Processing whole-slide images (WSIs) is highly complex, performed by pipelines that are often specifically adapted to individual patient cohorts. For reproducibility, we include the preprocessing functions for hemotoxylin and eosin (H\&E) and multiplexed immunofluorescence (mIF) slides in `preproc/process_hne.py` and `preproc/process_mif.py`. These are loosely based on the CLAM pipeline and involve thresholding via Otsu's method and subsequent embedding with an arbitrary pretrained model. The final output consists of three objects:

- `thumb`: A thumbnail of the image, by default downsampled by a factor of 256.
- `mask`: A binary mask indicating which of the patches were determined to be foreground and therefore to be embedded.
- `emb`: The image embeddings are as a $K \times M \times E$ tensor, where $K$ is the number of foreground patches, $M$ is the number of mIF  markers, and $E$ is the embedding dimension. In the case of H\&E slides, the tensors are of shape $K \times E$.

See the original manuscript for more details.

# Data loading

The [data module](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) is implemented by the `EmbeddedDataModule` class in `fluoroformer/data.py`. It expects a path to a directory `config_path` structured as follows:
```bash
config_path
├── SLIDE_1
│   ├── mask.npy
│   ├── thumb.npy
│   └── emb.pt
├── SLIDE_2
│   ├── mask.npy
│   ├── thumb.npy
│   └── emb.pt
├── ...
└── SLIDE_N
    ├── mask.npy
    ├── thumb.npy
    └── emb.pt
```
Note that `emb.pt` is expected to consist of a triple `(emb, time_, deceased)`, where `emb` is the output of the embedding script, `time_` is a scalar corresponding to the follow-up time, and `deceased` is Boolean indicating survival status.

The `train`, `val`, and `test` splits should be located in a yaml file that is passed to to the `EmbeddedDataModule` class. See the corresponding docstring for more detail.
