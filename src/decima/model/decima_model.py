from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import wandb
from grelu.model.heads import ConvHead
from grelu.model.models import BaseModel, BorzoiModel
from torch import nn


class DecimaModel(BaseModel):
    """
    Decima model.

    Args:
        n_tasks: Number of tasks.
        mask: Whether to use a mask.
        borzoi_kwargs: Keyword arguments for the Borzoi model.
    """

    def __init__(self, n_tasks: int, mask=True, borzoi_kwargs: dict = None, init_borzoi=False, replicate=0):
        borzoi_kwargs = {
            "crop_len": 5120,
            "n_tasks": 7611,
            "stem_channels": 512,
            "stem_kernel_size": 15,
            "init_channels": 608,
            "n_conv": 7,
            "kernel_size": 5,
            "n_transformers": 8,
            "key_len": 64,
            "value_len": 192,
            "pos_dropout": 0.0,
            "attn_dropout": 0.0,
            "n_heads": 8,
            "n_pos_features": 32,
            # backward compatibility with grelu<1.0.7
            "norm_kwargs": {"eps": 1e-5},
            "act_func": "gelu",
            "final_act_func": None,
            "final_pool_func": None,
            **(borzoi_kwargs or dict()),
        }
        model = BorzoiModel(**borzoi_kwargs)

        if model in ["0", "1", "2", "3"]:  # replicate index
            model = int(model)

        if init_borzoi:
            # Load state dict
            if Path(str(replicate)).exists():
                if replicate.endswith(".h5") or replicate.endswith(".pth") or replicate.endswith(".pt"):
                    state_dict = torch.load(replicate)
                elif replicate.endswith(".ckpt"):
                    state_dict = torch.load(replicate)["state_dict"]
                else:
                    raise ValueError(f"Invalid replicate path: {replicate}")
            else:
                wandb.login(host="https://api.wandb.ai/", anonymous="must")
                api = wandb.Api(overrides={"base_url": "https://api.wandb.ai/"})
                art = api.artifact(f"grelu/borzoi/human_state_dict_fold{replicate}:latest")
                with TemporaryDirectory() as d:
                    art.download(d)
                    state_dict = torch.load(Path(d) / f"fold{replicate}.h5")

            model.load_state_dict(state_dict)

        head = ConvHead(n_tasks=n_tasks, in_channels=1920, pool_func="avg")
        super().__init__(embedding=model.embedding, head=head)
        # Add a channel for the gene mask
        self.mask = mask
        if self.mask:
            weight = self.embedding.conv_tower.blocks[0].conv.weight
            new_layer = nn.Conv1d(5, 512, kernel_size=(15,), stride=(1,), padding="same")
            new_weight = nn.Parameter(torch.cat([weight, new_layer.weight[:, [-1], :]], axis=1))
            self.embedding.conv_tower.blocks[0].conv.weight = new_weight
