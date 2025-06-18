from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import wandb
from grelu.model.heads import ConvHead
from grelu.model.models import BaseModel, BorzoiModel
from torch import nn


class DecimaMLPHead(nn.Module):
    """
    Args:
        n_tasks: Number of tasks (output channels)
        norm: If True, batch normalization will be included.
        hidden_size: A list of dimensions for each hidden layer of the MLP.
        dropout: Dropout probability for the linear layers.
    """

    def __init__(
        self,
        n_tasks: int,
        len_pools: int = 4,
        hidden_size: List[int] = [],
        norm: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Save params
        self.n_tasks = n_tasks

        # Set up pool
        assert 6144 % len_pools = 0
        self.pool = nn.AvgPool1d(kernel_size=6144//len_pools)# B, 1920, 6144 -> B, 1920, len_pools

        # Set up MLP
        self.mlp = MLPHead(
            n_tasks=n_tasks,
            in_channels=1920,
            in_len=len_pools,
            act_func='relu',
            hidden_size = hidden_size,
            norm = norm,
            dropout = dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input data.
        """
        # Average over length axis
        x = self.pool(x) # N, 1920, len_pools
        x = self.mlp(x) # N, n_tasks, 1
        return x


class DecimaModel(BaseModel):
    def __init__(self, n_tasks: int, replicate: int = 0, mask=True, init_borzoi=True, head='conv', hidden_size=[1920], dropout=0.2, len_pools=1):
        self.mask = mask
        self.head=head
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

        if init_borzoi:
            # Load state dict
            wandb.login(host="https://api.wandb.ai/", anonymous="must")
            api = wandb.Api(overrides={"base_url": "https://api.wandb.ai/"})
            art = api.artifact(f"grelu/borzoi/human_state_dict_fold{replicate}:latest")
            with TemporaryDirectory() as d:
                art.download(d)
                state_dict = torch.load(Path(d) / f"fold{replicate}.h5")
            model.load_state_dict(state_dict)

        # Change head
        if self.head=='conv':
            head = ConvHead(n_tasks=n_tasks, in_channels=1920, pool_func="avg")

        elif self.head=='mlp':
            head = DecimaMLPHead(n_tasks=n_tasks, hidden_size=hidden_size, dropout=dropout, len_pools=len_pools)


        super().__init__(embedding=model.embedding, head=head)

        # Add a channel for the gene mask
        if self.mask:
            weight = self.embedding.conv_tower.blocks[0].conv.weight
            new_layer = nn.Conv1d(5, 512, kernel_size=(15,), stride=(1,), padding="same")
            new_weight = nn.Parameter(torch.cat([weight, new_layer.weight[:, [-1], :]], axis=1))
            self.embedding.conv_tower.blocks[0].conv.weight = new_weight
