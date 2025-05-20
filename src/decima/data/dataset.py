import warnings
import torch
import numpy as np
import pandas as pd
import h5py
import bioframe
from torch.utils.data import Dataset
from grelu.data.augment import Augmenter, _split_overall_idx
from grelu.sequence.utils import reverse_complement

from decima.constants import DECIMA_CONTEXT_SIZE
from decima.data.read_hdf5 import index_genes, indices_to_one_hot, _extract_center, mutate
from decima.core.result import DecimaResult


class HDF5Dataset(Dataset):
    def __init__(
        self,
        key,
        h5_file,
        ad=None,
        seq_len=DECIMA_CONTEXT_SIZE,
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
        seq_len=DECIMA_CONTEXT_SIZE,
        max_seq_shift=0,
        test_ref=False,
        include_cols=None,
        min_from_end=0,
        max_dist_tss=float("inf"),
    ):
        super().__init__()

        self.seq_len = seq_len
        self.h5_file = h5_file
        self.result = self._load_metadata(h5_file, ad)

        self.variants = self._overlap_genes(
            variants, include_cols=include_cols, min_from_end=min_from_end, max_dist_tss=max_dist_tss
        )
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

    @staticmethod
    def _load_metadata(h5_file=None, ad=None):
        if ad is None:
            return DecimaResult.load(h5_file)
        else:
            if h5_file is not None:
                warnings.warn("h5_file is ignored when ad is provided")
            return DecimaResult(ad)

    @staticmethod
    def overlap_genes(df_variants, df_genes, include_cols=None, min_from_end=0, max_dist_tss=float("inf")):
        df_variants = df_variants.copy()
        df_variants["start"] = df_variants.pos
        df_variants["end"] = df_variants["start"] + 1

        df = bioframe.overlap(df_genes, df_variants, how="inner", suffixes=("_gene", ""))

        df["rel_pos"] = df.apply(
            lambda row: row.pos - row.start_gene if row.strand_gene == "+" else row.end_gene - row.pos,
            axis=1,
        )
        df["rel_pos_end"] = df.apply(
            lambda row: row.end_gene - row.pos if row.strand_gene == "+" else row.pos - row.start_gene,
            axis=1,
        )
        df["ref_tx"] = df.apply(
            lambda row: row.ref if row.strand_gene == "+" else reverse_complement(row.ref),
            axis=1,
        )
        df["alt_tx"] = df.apply(
            lambda row: row.alt if row.strand_gene == "+" else reverse_complement(row.alt),
            axis=1,
        )
        # Get distance from TSS
        df["tss_dist"] = df.rel_pos - df.gene_mask_start_gene

        df = df[(df.rel_pos > min_from_end) & (df.rel_pos_end > min_from_end) & (df.tss_dist.abs() < max_dist_tss)]

        return df[
            [
                "chrom",
                "pos",
                "ref",
                "alt",
                "gene_gene",
                "start_gene",
                "end_gene",
                "strand_gene",
                "gene_mask_start_gene",
                "rel_pos",
                "ref_tx",
                "alt_tx",
                "tss_dist",
                *(include_cols or list()),
            ]
        ].rename(
            columns={
                "gene_gene": "gene",
                "start_gene": "start",
                "end_gene": "end",
                "strand_gene": "strand",
                "gene_mask_start_gene": "gene_mask_start",
            }
        )

    def _overlap_genes(self, variants, include_cols=None, min_from_end=0, max_dist_tss=float("inf")):
        return self.overlap_genes(
            variants,
            self.result.gene_metadata.reset_index(names=["gene"]),
            include_cols=include_cols,
            min_from_end=min_from_end,
            max_dist_tss=max_dist_tss,
        )

    def __len__(self):
        return self.n_seqs * self.n_augmented * self.n_alleles

    def close(self):
        self.dataset.close()

    def __getitem__(self, idx):
        seq_idx, augment_idx, allele_idx = _split_overall_idx(idx, (self.n_seqs, self.n_augmented, self.n_alleles))

        variant = self.variants.iloc[seq_idx]
        seq, mask = self.result.prepare_one_hot(variant.gene)

        if self.test_ref:  # check that ref is actually present
            assert ["A", "C", "G", "T"][seq[:4, variant.rel_pos].argmax()] == variant.ref_tx, (
                variant.ref_tx + "_vs_" + seq[:4, variant.rel_pos] + "__" + str(seq_idx)
            )

        if allele_idx:
            seq = mutate(seq, variant.alt_tx, variant.rel_pos)
        else:
            seq = mutate(seq, variant.ref_tx, variant.rel_pos)

        input = torch.vstack([seq, mask])
        # Augment the sequence
        input = _extract_center(input, seq_len=self.padded_seq_len)
        input = self.augmenter(seq=input, idx=augment_idx)
        return input
