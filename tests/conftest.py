import torch
import pytest
from decima.hub import login_wandb


login_wandb()


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
