import torch
import numpy as np
import pandas as pd
import h5py
import bioframe
from more_itertools import flatten
from torch.utils.data import Dataset, default_collate
from grelu.data.augment import Augmenter, _split_overall_idx
from grelu.sequence.utils import reverse_complement

from decima.constants import DECIMA_CONTEXT_SIZE
from decima.data.read_hdf5 import index_genes, indices_to_one_hot, _extract_center
from decima.core.result import DecimaResult

from decima.model.metrics import WarningType


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
        anndata=None,
        seq_len=DECIMA_CONTEXT_SIZE,
        max_seq_shift=0,
        include_cols=None,
        gene_col=None,
        min_from_end=0,
        max_distance=float("inf"),
        max_distance_type="tss",
    ):
        super().__init__()

        self.seq_len = seq_len
        self.result = DecimaResult.load(anndata)

        self.variants = self._overlap_genes(
            variants,
            include_cols=include_cols,
            gene_col=gene_col,
            min_from_end=min_from_end,
            max_distance=max_distance,
            max_distance_type=max_distance_type,
        )

        self.n_seqs = len(self.variants)
        self.n_alleles = 2

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
    def overlap_genes(
        df_variants,
        df_genes,
        gene_col=None,
        include_cols=None,
        min_from_end=0,
        max_distance=float("inf"),
        max_distance_type="tss",
    ):
        include_cols = include_cols or list()

        df_variants = df_variants.copy()
        df_variants["start"] = df_variants.pos
        df_variants["end"] = df_variants["start"] + 1

        if gene_col is not None:
            assert gene_col in df_variants.columns, f"Gene column {gene_col} not found in variants"

            missing = df_variants[~df_variants[gene_col].isin(set(df_genes["gene"]))]
            if len(missing) > 0:
                raise ValueError(f"Some genes in {gene_col} are not in the result: {missing[gene_col].unique()}")

            df_variants = df_variants.rename(columns={gene_col: "gene"})
            df = df_variants.merge(df_genes, how="left", on="gene", suffixes=("", "_gene"))
        else:
            df = bioframe.overlap(df_genes, df_variants, how="inner", suffixes=("_gene", ""))

        df = df.rename(
            columns={
                "start": "start_variant",
                "end": "end_variant",
                "gene_gene": "gene",
                "start_gene": "start",
                "end_gene": "end",
                "strand_gene": "strand",
                "gene_mask_start_gene": "gene_mask_start",
                "gene_mask_end_gene": "gene_mask_end",
            }
        )

        df["rel_pos"] = df.apply(
            lambda row: row.pos - row.start if row.strand == "+" else row.end - row.pos,
            axis=1,
        )
        df["rel_pos_end"] = df.apply(
            lambda row: row.end - row.pos if row.strand == "+" else row.pos - row.start,
            axis=1,
        )
        df["ref_tx"] = df.apply(
            lambda row: row.ref if row.strand == "+" else reverse_complement(row.ref),
            axis=1,
        )
        df["alt_tx"] = df.apply(
            lambda row: row.alt if row.strand == "+" else reverse_complement(row.alt),
            axis=1,
        )
        df["tss_dist"] = df.rel_pos - df.gene_mask_start

        df = df[(df.rel_pos > min_from_end) & (df.rel_pos_end > min_from_end)]  # & (df.tss_dist.abs() < max_dist_tss)

        if max_distance_type == "tss":
            df = df[df.tss_dist.abs() < max_distance]
        elif max_distance_type == "gene":
            raise NotImplementedError("max_distance_type == 'gene' is not implemented yet.")
        else:
            raise ValueError(f"Invalid max_distance_type: {max_distance_type}. Must be 'tss'.")

        return df[
            [
                "chrom",
                "pos",
                "ref",
                "alt",
                "gene",
                "start",
                "end",
                "strand",
                "gene_mask_start",
                "gene_mask_end",
                "rel_pos",
                "ref_tx",
                "alt_tx",
                "tss_dist",
                *include_cols,
            ]
        ]

    def _overlap_genes(
        self,
        variants,
        include_cols=None,
        gene_col=None,
        min_from_end=0,
        max_distance=float("inf"),
        max_distance_type="tss",
    ):
        return self.overlap_genes(
            variants,
            self.result.gene_metadata.reset_index(names=["gene"]),
            include_cols=include_cols,
            gene_col=gene_col,
            min_from_end=min_from_end,
            max_distance=max_distance,
            max_distance_type=max_distance_type,
        )

    def __len__(self):
        return self.n_seqs * self.n_augmented * self.n_alleles

    def validate_allele_seq(self, gene, variant):
        seq = self.result.gene_sequence(gene)
        return (seq[variant.rel_pos] == variant.ref_tx) or (seq[variant.rel_pos] == variant.alt_tx)

    def __getitem__(self, idx):
        seq_idx, augment_idx, allele_idx = _split_overall_idx(idx, (self.n_seqs, self.n_augmented, self.n_alleles))

        variant = self.variants.iloc[seq_idx]

        warnings = list()
        if not self.validate_allele_seq(variant.gene, variant):
            warnings.append(WarningType.ALLELE_MISMATCH_WITH_REFERENCE_GENOME)

        if allele_idx:
            seq, mask = self.result.prepare_one_hot(
                variant.gene,
                variants=[{"chrom": variant.chrom, "pos": variant.pos, "ref": variant.ref, "alt": variant.alt}],
            )
        else:
            seq, mask = self.result.prepare_one_hot(
                variant.gene,
                variants=[{"chrom": variant.chrom, "pos": variant.pos, "ref": variant.alt, "alt": variant.ref}],
            )

        inputs = torch.vstack([seq, mask])

        inputs = _extract_center(inputs, seq_len=self.padded_seq_len)
        inputs = self.augmenter(seq=inputs, idx=augment_idx)
        return {
            "seq": inputs,
            "warning": warnings,
        }

    def collate_fn(self, batch):
        return {
            "seq": default_collate([i["seq"] for i in batch]),
            "warning": list(flatten([b["warning"] for b in batch])),
        }
