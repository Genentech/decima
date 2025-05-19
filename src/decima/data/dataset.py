import torch
import numpy as np
import pandas as pd
import h5py
from torch.utils.data import Dataset
from grelu.data.augment import Augmenter, _split_overall_idx

from decima.data.read_hdf5 import indices_to_one_hot, _extract_center, get_gene_idx, mutate
from decima.core.result import DecimaResult


class HDF5Dataset(Dataset):
    def __init__(
        self,
        key,
        h5_file,
        ad=None,
        seq_len=524288,
        max_seq_shift=0,
        seed=0,
        augment_mode="random",
    ):
        super().__init__()

        # Save data params
        self.h5_file = h5_file
        self.seq_len = seq_len
        self.key = key

        # Save augmentation params
        self.max_seq_shift = max_seq_shift
        self.augmenter = Augmenter(
            rc=False,
            max_seq_shift=self.max_seq_shift,
            max_pair_shift=0,
            seq_len=self.seq_len,
            label_len=None,
            seed=seed,
            mode=augment_mode,
        )
        self.n_augmented = len(self.augmenter)
        self.padded_seq_len = self.seq_len + (2 * self.max_seq_shift)

        # Index genes
        self.gene_index = index_genes(self.h5_file, key=self.key)
        self.n_seqs = len(self.gene_index)

        # Setup
        self.dataset = h5py.File(self.h5_file, "r")
        self.extract_tasks(ad)
        self.predict = False
        self.n_alleles = 1

    def __len__(self):
        return self.n_seqs * self.n_augmented

    def close(self):
        self.dataset.close()

    def extract_tasks(self, ad=None):
        tasks = np.array(self.dataset["tasks"]).astype(str)
        if ad is not None:
            assert np.all(tasks == ad.obs_names)
            self.tasks = ad.obs
        else:
            self.tasks = pd.DataFrame(index=tasks)

    def extract_seq(self, idx):
        seq = self.dataset["sequences"][idx]
        seq = indices_to_one_hot(seq)  # 4, L
        mask = self.dataset["masks"][[idx]]  # 1, L
        seq = np.concatenate([seq, mask])  # 5, L
        seq = _extract_center(seq, seq_len=self.padded_seq_len)
        return torch.Tensor(seq)

    def extract_label(self, idx):
        return torch.Tensor(self.dataset["labels"][idx])

    def __getitem__(self, idx):
        # Augment
        seq_idx, augment_idx = _split_overall_idx(idx, (self.n_seqs, self.n_augmented))

        # Extract the sequence
        gene_idx = self.gene_index[seq_idx]
        seq = self.extract_seq(gene_idx)

        # Augment the sequence
        seq = self.augmenter(seq=seq, idx=augment_idx)

        if self.predict:
            return seq

        else:
            label = self.extract_label(gene_idx)
            return seq, label


class VariantDataset(Dataset):
    def __init__(
        self,
        variants,
        h5_file=None,
        ad=None,
        seq_len=524288,
        max_seq_shift=0,
        test_ref=False,
    ):
        super().__init__()

        # Save data params
        self.seq_len = seq_len
        self.h5_file = h5_file

        # Save variant params
        self.variants = variants[["gene", "rel_pos", "ref_tx", "alt_tx"]].copy()
        self.n_seqs = len(self.variants)
        self.n_alleles = 2
        self.test_ref = test_ref

        # Save augmentation params
        self.max_seq_shift = max_seq_shift
        self.augmenter = Augmenter(
            rc=False,
            max_seq_shift=self.max_seq_shift,
            max_pair_shift=0,
            seq_len=self.seq_len,
            label_len=None,
            mode="serial",
        )
        self.n_augmented = len(self.augmenter)
        self.padded_seq_len = self.seq_len + (2 * self.max_seq_shift)

        # Map each variant to the corresponding gene in the h5 file
        if ad is None:
            assert self.h5_file is not None
            gene_map = {gene: get_gene_idx(self.h5_file, gene) for gene in self.variants.gene.unique()}
            self.dataset = h5py.File(self.h5_file, "r")
            self.pad = self.dataset["pad"]
        else:
            gene_map = {gene: i for i, gene in enumerate(self.variants.gene.unique())}
            result = DecimaResult(ad)
            seqs, masks = list(zip(*[result.prepare_one_hot(gene) for gene in gene_map.keys()]))
            self.dataset = {
                "sequences": torch.stack(seqs),
                "masks": torch.vstack(masks),
            }
            self.pad = 0

        # Setup
        self.variants["gene_idx"] = self.variants.gene.map(gene_map)

    def __len__(self):
        return self.n_seqs * self.n_augmented * self.n_alleles

    def close(self):
        self.dataset.close()

    def extract_seq(self, idx):
        seq = self.dataset["sequences"][idx]
        if self.h5_file is not None:
            seq = indices_to_one_hot(seq)  # 4, L
        mask = self.dataset["masks"][[idx]]  # 1, L
        seq = np.concatenate([seq, mask])  # 5, L
        return torch.Tensor(seq)

    def __getitem__(self, idx):
        # Get indices
        seq_idx, augment_idx, allele_idx = _split_overall_idx(idx, (self.n_seqs, self.n_augmented, self.n_alleles))

        # Extract the sequence
        variant = self.variants.iloc[seq_idx]
        seq = self.extract_seq(int(variant.gene_idx))

        if self.test_ref:  # check that ref is actually present
            assert ["A", "C", "G", "T"][seq[:4, variant.rel_pos + self.pad].argmax()] == variant.ref_tx, (
                variant.ref_tx + "_vs_" + seq[:4, variant.rel_pos + self.pad] + "__" + str(seq_idx)
            )

        # Insert the allele
        if allele_idx:
            seq = mutate(seq, variant.alt_tx, variant.rel_pos + self.pad)
        else:
            seq = mutate(seq, variant.ref_tx, variant.rel_pos + self.pad)

        # Augment the sequence
        seq = _extract_center(seq, seq_len=self.padded_seq_len)
        seq = self.augmenter(seq=seq, idx=augment_idx)
        return seq
