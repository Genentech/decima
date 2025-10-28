import torch
import numpy as np
from grelu.sequence.format import convert_input_type

from decima.constants import DECIMA_CONTEXT_SIZE


def prepare_mask_gene(gene_start, gene_end, padding=0):
    """Prepare gene mask tensor for gene regions.

    Args:
        gene_start: Start position of the gene
        gene_end: End position of the gene
        padding: Amount of padding to add on both sides. Defaults to 0

    Returns:
        torch.Tensor: Gene mask tensor with 1s in gene region and 0s elsewhere
    """
    mask = np.zeros(shape=(1, DECIMA_CONTEXT_SIZE + padding * 2))
    mask[0, gene_start:gene_end] += 1
    return torch.from_numpy(mask).float()


def one_hot_to_seq(one_hot):
    """Convert one-hot encoded sequence to a string

    Args:
        one_hot (np.ndarray or torch.Tensor): One-hot encoded sequence

    Returns:
        str: String representation of the sequence
    """
    if isinstance(one_hot, np.ndarray):
        one_hot = torch.from_numpy(one_hot)
    elif isinstance(one_hot, torch.Tensor):
        pass
    else:
        raise ValueError(f"Invalid type for one_hot: {type(one_hot)}. Must be a numpy array or torch tensor.")

    if len(one_hot.shape) == 2:
        one_hot = one_hot[:4, :]
    elif len(one_hot.shape) == 3:
        one_hot = one_hot[:, :4, :]
    else:
        raise ValueError(f"Invalid shape for one_hot: {one_hot.shape}. Must be 2-dimensional or 3-dimensional.")

    return convert_input_type(one_hot, "strings", input_type="one_hot")
