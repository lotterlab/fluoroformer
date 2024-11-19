import signal
import traceback
import sys
from fluoroformer import Learner, EmbeddedDataModule
from pytorch_lightning.cli import LightningCLI

def main():
    cli = LightningCLI(
        Learner, EmbeddedDataModule, save_config_kwargs={"overwrite": True}
    )

if __name__ == "__main__":
    main()

