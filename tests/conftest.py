import torch
import pytest


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
