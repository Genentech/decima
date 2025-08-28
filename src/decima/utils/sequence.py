import torch
import numpy as np

from decima.constants import DECIMA_CONTEXT_SIZE


def prepare_mask_gene(gene_start, gene_end, padding=0):
    """Mask a gene sequence with a padding.

    Args:
        gene_start: Start of the gene in the decima context window.
        gene_end: End of the gene in the decima context window.
        padding: Padding to add to the gene mask

    Returns:
        torch.Tensor: Masked gene sequence with shape (1, DECIMA_CONTEXT_SIZE + padding * 2)
    """
    mask = np.zeros(shape=(1, DECIMA_CONTEXT_SIZE + padding * 2))
    mask[0, gene_start:gene_end] += 1
    return torch.from_numpy(mask).float()
