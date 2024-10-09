import torch
from torch import nn
from grelu.model.models import BorzoiModel, BaseModel
from grelu.model.heads import ConvHead
from grelu.resources import get_artifact
from tempfile import TemporaryDirectory
from pathlib import Path
import wandb


class DecimaModel(BaseModel):

    def __init__(self, n_tasks: int, replicate: int = 0, mask=True):
        self.mask = mask
        model = BorzoiModel(
            crop_len=5120,
            n_tasks=7611,
            stem_channels=512,
            stem_kernel_size=15,
            init_channels=608,
            n_conv=7,
            kernel_size=5,
            n_transformers=8,
            key_len=64,
            value_len=192,
            pos_dropout=0.0,
            attn_dropout=0.0,
            n_heads=8,
            n_pos_features=32,
            final_act_func=None,
            final_pool_func=None,
        )

        # Load state dict
        api = wandb.Api(overrides={'base_url':"https://genentech.wandb.io"})
        art = api.artifact(f'grelu/borzoi/human_state_dict_fold{replicate}:latest')
        with TemporaryDirectory() as d:
            art.download(d)
            state_dict = torch.load(Path(d) / f"fold{replicate}.h5")
        model.load_state_dict(state_dict)

        # Change head
        head = ConvHead(n_tasks=n_tasks, in_channels=1920, pool_func="avg")

        super().__init__(embedding=model.embedding, head=head)

        # Add a channel for the gene mask
        if self.mask:
            weight = self.embedding.conv_tower.blocks[0].conv.weight
            new_layer = nn.Conv1d(
                5, 512, kernel_size=(15,), stride=(1,), padding="same"
            )
            new_weight = nn.Parameter(
                torch.cat([weight, new_layer.weight[:, [-1], :]], axis=1)
            )
            self.embedding.conv_tower.blocks[0].conv.weight = new_weight
