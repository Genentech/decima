import h5py
import numpy as np
import torch
from grelu.sequence.format import BASE_TO_INDEX_HASH, indices_to_one_hot


def count_genes(h5_file, key=None):
    with h5py.File(h5_file, "r") as f:
        genes = np.array(f["genes"]).astype(str)
    if key is None:
        return genes.shape[0]
    else:
        return np.sum(genes[:, 1] == key)


def index_genes(h5_file, key=None):
    with h5py.File(h5_file, "r") as f:
        genes = np.array(f["genes"]).astype(str)
    if key is None:
        return np.array(range(len(genes)))
    else:
        return np.where(genes[:, 1] == key)[0]


def list_genes(h5_file, key=None):
    with h5py.File(h5_file, "r") as f:
        genes = np.array(f["genes"]).astype(str)
    if key is None:
        return genes[:, 0]
    else:
        return genes[genes[:, 1] == key, 0]


def get_gene_idx(h5_file, gene, key=None):
    gene_ord = list_genes(h5_file, key=None)
    return np.where(gene_ord == gene)[0][0]


def _extract_center(x, seq_len, shift=0):
    start = (x.shape[-1] - seq_len) // 2
    start -= shift
    return x[..., start : start + seq_len]


def extract_gene_data(h5_file, gene, seq_len=524288, merge=True):
    gene_idx = get_gene_idx(h5_file, key=None, gene=gene)

    with h5py.File(h5_file, "r") as f:
        seq = np.array(f["sequences"][gene_idx])
        seq = indices_to_one_hot(seq)
        mask = torch.Tensor(np.array(f["masks"][[gene_idx]]))

    seq = _extract_center(seq, seq_len=seq_len)
    mask = _extract_center(mask, seq_len=seq_len)

    if merge:
        return torch.vstack([seq, mask])
    else:
        return seq, mask


def mutate(seq, allele, pos):
    idx = BASE_TO_INDEX_HASH[allele]
    seq[:4, pos] = 0
    seq[idx, pos] = 1
    return seq
