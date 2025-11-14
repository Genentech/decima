"""
pytorch Datasets for Decima.

This module contains the datasets for Decima, including:
- HDF5Dataset: Dataset for HDF5 files.
- GeneDataset: Dataset for gene expression prediction.
- SeqDataset: Dataset for sequence prediction.
- VariantDataset: Dataset for variant effect prediction.
"""

from typing import List, Optional
import warnings
import torch
import h5py
import bioframe
import numpy as np
import pandas as pd
from more_itertools import flatten
from torch.utils.data import Dataset, default_collate
from grelu.sequence.format import indices_to_strings, indices_to_one_hot
from grelu.data.augment import Augmenter, _split_overall_idx
from grelu.sequence.format import strings_to_one_hot
from grelu.sequence.utils import reverse_complement

from decima.constants import DECIMA_CONTEXT_SIZE, ENSEMBLE_MODELS, MODEL_METADATA
from decima.data.read_hdf5 import _extract_center, index_genes
from decima.core.result import DecimaResult
from decima.utils.io import read_fasta_gene_mask
from decima.utils.sequence import prepare_mask_gene, one_hot_to_seq

from decima.model.metrics import WarningType


class HDF5Dataset(Dataset):
    """
    Dataset for HDF5 files.

    Args:
        key: Key to use to access the data in the HDF5 file.
        h5_file: Path to the HDF5 file.
        ad: AnnData object to use for extracting tasks.
        seq_len: Length of the sequence.
        max_seq_shift: Maximum sequence shift.
        seed: Seed for the random number generator.
        augment_mode: Augmentation mode.

    Returns:
        Dataset: Dataset for HDF5 files.

    """

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

        # Setup - Open file and cache data needed for worker processes
        self.dataset = None
        self._is_closed = False
        self._open_file()
        self.extract_tasks(ad)
        self.predict = False
        self.n_alleles = 1

    def _open_file(self):
        """Open the HDF5 file. This will be called in each worker process."""
        if self.dataset is None or self._is_closed:
            self.dataset = h5py.File(self.h5_file, "r")
            self._is_closed = False

    def __len__(self):
        return self.n_seqs * self.n_augmented

    def close(self):
        if self.dataset is not None and not self._is_closed:
            self.dataset.close()
            self._is_closed = True

    def extract_tasks(self, ad=None):
        self._open_file()
        tasks = np.array(self.dataset["tasks"]).astype(str)
        if ad is not None:
            assert np.all(tasks == ad.obs_names)
            self.tasks = ad.obs
        else:
            self.tasks = pd.DataFrame(index=tasks)

    def extract_seq(self, idx):
        self._open_file()
        seq = self.dataset["sequences"][idx]
        seq = indices_to_one_hot(seq)  # 4, L
        mask = self.dataset["masks"][[idx]]  # 1, L
        seq = np.concatenate([seq, mask])  # 5, L
        seq = _extract_center(seq, seq_len=self.padded_seq_len)
        return torch.Tensor(seq)

    def extract_label(self, idx):
        self._open_file()
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


class GeneDataset(Dataset):
    """
    Dataset for gene expression prediction.

    Args:
        genes: List of genes to include in the dataset.
        metadata_anndata: AnnData object to use for extracting gene metadata.
        max_seq_shift: Maximum sequence shift.
        seed: Seed for the random number generator.
        augment_mode: Augmentation mode.
        genome: Name of the genome

    Returns:
        Dataset: Dataset for gene expression prediction.

    Examples:
        >>> genes = [
        ...     "SPI1",
        ...     "SPI2",
        ... ]
        >>> dataset = (
        ...     GeneDataset(
        ...         genes=genes
        ...     )
        ... )
        >>> dl = torch.data.DataLoader(
        ...     dataset,
        ...     batch_size=1,
        ...     shuffle=True,
        ...     collate_fn=dataset.collate_fn,
        ... )
        >>> for batch in dl:
            print(batch)
        ... (2, 524288, 5)
    """

    def __init__(
        self,
        genes=None,
        metadata_anndata: Optional[str] = None,
        max_seq_shift: int = 0,
        seed: int = 0,
        augment_mode: str = "random",
        genome: str = "hg38",
    ):
        super().__init__()

        self.genome = genome
        self.result = DecimaResult.load(metadata_anndata)
        self.genes = genes or list(self.result.genes)
        self.gene_mask_starts = self.result.gene_metadata.loc[self.genes, "gene_mask_start"].values
        self.gene_mask_ends = self.result.gene_metadata.loc[self.genes, "gene_mask_end"].values

        # Save augmentation params
        self.max_seq_shift = max_seq_shift
        self.augment_mode = augment_mode
        self.augmenter = Augmenter(
            rc=False,
            max_seq_shift=self.max_seq_shift,
            max_pair_shift=0,
            seq_len=DECIMA_CONTEXT_SIZE,
            label_len=None,
            seed=seed,
            mode=augment_mode,
        )
        self.n_augmented = len(self.augmenter)
        self.padded_seq_len = DECIMA_CONTEXT_SIZE + (2 * self.max_seq_shift)

        self.n_seqs = len(self.genes) * self.n_augmented
        self.n_alleles = 1
        self.predict = False

    def __len__(self):
        return self.n_seqs

    def __getitem__(self, idx):
        seq_idx, augment_idx = _split_overall_idx(idx, (self.n_seqs, self.n_augmented))

        seq, mask = self.result.prepare_one_hot(self.genes[seq_idx], padding=self.max_seq_shift, genome=self.genome)
        inputs = torch.vstack([seq, mask])
        inputs = self.augmenter(seq=inputs, idx=augment_idx)

        return inputs

    def collate_fn(self, batch):
        return default_collate(batch)


class SeqDataset(Dataset):
    """
    Dataset for sequence prediction with the masked gene sequence.

    Args:
        seqs: List of sequences as strings.
        gene_mask_starts: List of gene mask starts.
        gene_mask_ends: List of gene mask ends.
        genes: List of gene names.
        max_seq_shift: Maximum sequence shift.
        seed: Seed for the random number generator.
        augment_mode: Augmentation mode.

    Returns:
        Dataset: Dataset for sequence prediction with the masked gene sequence.

    Examples:
        >>> seqs = [
        ...     "ATCG...",
        ...     "ATCG..",
        ...     "ATCG...",
        ... ]
        >>> gene_mask_starts = [
        ...     0,
        ...     0,
        ...     0,
        ... ]
        >>> gene_mask_ends = [
        ...     4,
        ...     4,
        ...     4,
        ... ]
        >>> dataset = SeqDataset(
        ...     seqs=seqs,
        ...     gene_mask_starts=gene_mask_starts,
        ...     gene_mask_ends=gene_mask_ends,
        ... )
        >>> dl = torch.data.DataLoader(
        ...     dataset,
        ...     batch_size=1,
        ...     shuffle=True,
        ...     collate_fn=dataset.collate_fn,
        ... )
        >>> for batch in dl:
            print(batch)
        ... (2, 524288, 5)

        >>> dataset = SeqDataset.from_fasta(
        ...     fasta_file="example/seqs.fasta"
        ... )

        >>> df = pd.DataFrame(
        ...     {
        ...         "seq": [
        ...             "ATCG..",
        ...             "ATCG...",
        ...             "ATCG......",
        ...         ],
        ...         "gene_mask_start": [
        ...             0,
        ...             0,
        ...             0,
        ...         ],
        ...         "gene_mask_end": [
        ...             4,
        ...             4,
        ...             4,
        ...         ],
        ...     }
        ... )
        >>> dataset = SeqDataset.from_dataframe(
        ...     df
        ... )
    """

    def __init__(
        self,
        seqs: List[str],
        gene_mask_starts: List[int],
        gene_mask_ends: List[int],
        genes: List[str] = None,
        max_seq_shift: int = 0,
        seed: int = 0,
        augment_mode: str = "random",
    ):
        assert (
            len(seqs) == len(gene_mask_starts) == len(gene_mask_ends)
        ), "Lengths of `seqs`, `gene_mask_starts`, and `gene_mask_ends` must match."

        self.seqs = seqs
        self.gene_mask_starts = gene_mask_starts
        self.gene_mask_ends = gene_mask_ends

        # Save augmentation params
        self.max_seq_shift = max_seq_shift
        self.augment_mode = augment_mode
        self.augmenter = Augmenter(
            rc=False,
            max_seq_shift=self.max_seq_shift,
            max_pair_shift=0,
            seq_len=DECIMA_CONTEXT_SIZE,
            label_len=None,
            seed=seed,
            mode=augment_mode,
        )
        self.n_augmented = len(self.augmenter)
        self.padded_seq_len = DECIMA_CONTEXT_SIZE + (2 * self.max_seq_shift)

        self.n_seqs = len(self.seqs) * self.n_augmented
        self.n_alleles = 1
        self.predict = False

        if genes is None:
            self.genes = [f"seq{i}" for i in range(len(self.seqs))]
        else:
            assert len(genes) == len(self.seqs), "Length of `genes` must match length of `seqs`"
            self.genes = [str(i) for i in genes]

    def __len__(self):
        return self.n_seqs

    def __getitem__(self, idx):
        seq_idx, augment_idx = _split_overall_idx(idx, (self.n_seqs, self.n_augmented))
        seq = self.seqs[seq_idx]
        gene_mask_start = self.gene_mask_starts[seq_idx]
        gene_mask_end = self.gene_mask_ends[seq_idx]

        inputs = torch.vstack(
            [
                strings_to_one_hot(seq),
                prepare_mask_gene(gene_mask_start, gene_mask_end, padding=self.max_seq_shift),
            ]
        )
        inputs = self.augmenter(seq=inputs, idx=augment_idx)

        return inputs

    def collate_fn(self, batch):
        return default_collate(batch)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, max_seq_shift: int = 0, seed: int = 0, augment_mode: str = "random"):
        """
        Create a SeqDataset from a pandas DataFrame.

        Args:
            df: pandas DataFrame containing `seq`, `gene_mask_start`, and `gene_mask_end` columns.
            max_seq_shift: Maximum sequence shift.
            seed: Seed for the random number generator.
            augment_mode: Augmentation mode.

        Returns:
            SeqDataset: SeqDataset object.

        Examples:
            >>> df = pd.DataFrame(
            ...     {
            ...         "seq": [
            ...             "ATCG..",
            ...             "ATCG...",
            ...             "ATCG......",
            ...         ],
            ...         "gene_mask_start": [
            ...             0,
            ...             0,
            ...             0,
            ...         ],
            ...         "gene_mask_end": [
            ...             4,
            ...             4,
            ...             4,
            ...         ],
            ...     }
            ... )
            >>> dataset = SeqDataset.from_dataframe(
            ...     df
            ... )
            >>> dl = torch.data.DataLoader(
            ...     dataset,
            ...     batch_size=1,
            ...     shuffle=True,
            ...     collate_fn=dataset.collate_fn,
            ... )
            >>> for batch in dl:
                print(batch)
            ... (2, 524288, 5)
        """
        assert "seq" in df.columns, "`df` must contain `seq` column"
        assert "gene_mask_start" in df.columns, "`df` must contain `gene_mask_start` column"
        assert "gene_mask_end" in df.columns, "`df` must contain `gene_mask_end` column"

        return cls(
            df["seq"].tolist(),
            df["gene_mask_start"].tolist(),
            df["gene_mask_end"].tolist(),
            genes=df.index.tolist(),
            max_seq_shift=max_seq_shift,
            seed=seed,
            augment_mode=augment_mode,
        )

    @classmethod
    def from_fasta(cls, fasta_file: str, max_seq_shift: int = 0, seed: int = 0, augment_mode: str = "random"):
        """
        Create a SeqDataset from a FASTA file.

        Args:
            fasta_file: Path to the FASTA file with header as gene name, maks and sequence as the sequence: ">gene_name|gene_mask_start=10000|gene_mask_end=10000\nATACG...".
            max_seq_shift: Maximum sequence shift.
            seed: Seed for the random number generator.
            augment_mode: Augmentation mode.

        Returns:
            SeqDataset: SeqDataset object.

        Examples:
            >>> dataset = SeqDataset.from_fasta(
            ...     fasta_file="example/seqs.fasta"
            ... )
            >>> dl = torch.data.DataLoader(
            ...     dataset,
            ...     batch_size=1,
            ...     shuffle=True,
            ...     collate_fn=dataset.collate_fn,
            ... )
            >>> for batch in dl:
                print(batch)
            ... (2, 524288, 5)
        """
        seqs = read_fasta_gene_mask(fasta_file)
        return cls.from_dataframe(seqs, max_seq_shift=max_seq_shift, seed=seed, augment_mode=augment_mode)

    @classmethod
    def from_one_hot(
        cls,
        one_hot: torch.Tensor,
        gene_mask_starts: List[int] = None,
        gene_mask_ends: List[int] = None,
        max_seq_shift: int = 0,
        seed: int = 0,
        augment_mode: str = "random",
    ):
        """Create a SeqDataset from a one-hot encoded tensor.

        Args:
            one_hot: One-hot encoded tensor with shape (batch_size, 4 or 5, seq_len).
            gene_mask_starts: List of gene mask starts.
            gene_mask_ends: List of gene mask ends.
            max_seq_shift: Maximum sequence shift.
            seed: Seed for the random number generator.
            augment_mode: Augmentation mode.

        Returns:
            SeqDataset: SeqDataset object.
        """
        assert len(one_hot.shape) == 3, "`one_hot` must be 3-dimensional with shape (batch_size, 4 or 5, seq_len)"
        assert (
            one_hot.shape[2] == DECIMA_CONTEXT_SIZE
        ), "`one_hot` must have the same sequence length as DECIMA_CONTEXT_SIZE"
        seqs = one_hot_to_seq(one_hot)

        if (gene_mask_starts is None) and (gene_mask_ends is None):
            assert one_hot.shape[1] == 5, (
                "`seqs` must be 5-dimensional with shape (batch_size, 5, seq_len) "
                "where the 2th dimension is a one_hot encoded seq and binary mask gene mask."
            )
            gene_mask_starts = [int(torch.where(i[4] == 1)[0].min()) for i in one_hot]
            gene_mask_ends = [int(torch.where(i[4] == 1)[0].max()) for i in one_hot]

        elif (gene_mask_starts is not None) and (gene_mask_ends is not None):
            assert (
                len(gene_mask_starts) == len(gene_mask_ends) == len(one_hot)
            ), "Lengths of `gene_mask_starts`, `gene_mask_ends`, and `one_hot` must match."
        else:
            raise ValueError(
                "Either `gene_mask_starts` and `gene_mask_ends` must be provided or `one_hot` should contain gene masks."
            )
        return cls(
            seqs, gene_mask_starts, gene_mask_ends, max_seq_shift=max_seq_shift, seed=seed, augment_mode=augment_mode
        )

    def __str__(self):
        return f"SeqDataset(n_seqs={self.n_seqs}, n_augmented={self.n_augmented}, n_alleles={self.n_alleles})"


class VariantDataset(Dataset):
    """Dataset for variant effect prediction

    Args:
        variants (pd.DataFrame): DataFrame with variants
        anndata (AnnData): AnnData object with gene metadata
        seq_len (int): Length of the sequence
        max_seq_shift (int): Maximum sequence shift
        include_cols (list): List of columns to include in the output
        gene_col (str): Column name for gene names
        min_from_end (int): Minimum distance from the end of the gene
        distance_type (str): Type of distance
        min_distance (int): Minimum distance from the TSS
        max_distance (int): Maximum distance from the TSS

    Returns:
        Dataset: Dataset for variant effect prediction

    Examples:
        >>> import pandas as pd
        >>> import anndata as ad
        >>> from decima.data.dataset import (
        ...     VariantDataset,
        ... )
        >>> variants = pd.read_csv(
        ...     "variants.csv"
        ... )
        >>> dataset = (
        ...     VariantDataset(
        ...         variants
        ...     )
        ... )
        >>> dataset[0]
        {'seq': tensor([[1.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 1.0000],
                        [0.0000, 1.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
                        [0.0000, 0.0000, 1.0000,  ..., 0.0000, 1.0000, 0.0000],
                        [0.0000, 0.0000, 0.0000,  ..., 1.0000, 0.0000, 0.0000],
                        [0.0000, 0.0000, 1.0000,  ..., 1.0000, 0.0000, 0.0000]]), 'warning': []}
        >>> dl = torch.data.DataLoader(
        ...     dataset,
        ...     batch_size=1,
        ...     shuffle=True,
        ...     collate_fn=dataset.collate_fn,
        ... )
        >>> for batch in dl:
            print(batch)
        ... {
            'seq': tensor([[1.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 1.0000],
                            [0.0000, 1.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
                            [0.0000, 0.0000, 1.0000,  ..., 0.0000, 1.0000, 0.0000],
                            [0.0000, 0.0000, 1.0000,  ..., 1.0000, 0.0000, 0.0000]]),
            'warning': [],
            'pred_expression': tensor([[0.0000, 0.0000],
                            [0.0000, 0.0000],
                            [0.0000, 0.0000],
                            [0.0000, 0.0000]]),
        }
    """

    DEFAULT_COLUMNS = [
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
    ]

    def __init__(
        self,
        variants,
        metadata_anndata: Optional[str] = None,
        max_seq_shift: int = 0,
        seed: int = 0,
        include_cols=None,
        gene_col: Optional[str] = None,
        min_from_end: int = 0,
        distance_type: str = "tss",
        min_distance: int = 0,
        max_distance: float = float("inf"),
        model_name: Optional[str] = None,
        reference_cache: bool = True,
        genome: str = "hg38",
    ):
        super().__init__()

        self.reference_cache = reference_cache
        self.genome = genome
        self.result = DecimaResult.load(metadata_anndata or model_name)

        self.variants = self._overlap_genes(
            variants,
            include_cols=include_cols,
            gene_col=gene_col,
            min_from_end=min_from_end,
            distance_type=distance_type,
            min_distance=min_distance,
            max_distance=max_distance,
        )

        self.n_seqs = len(self.variants)
        self.n_alleles = 2

        # Save augmentation params
        self.max_seq_shift = max_seq_shift
        self.augmenter = Augmenter(
            rc=False,
            max_seq_shift=self.max_seq_shift,
            max_pair_shift=0,
            seq_len=DECIMA_CONTEXT_SIZE,
            label_len=None,
            seed=seed,
            mode="serial",
        )
        self.n_augmented = len(self.augmenter)
        self.padded_seq_len = DECIMA_CONTEXT_SIZE + (2 * self.max_seq_shift)

        if (model_name is None) or (not reference_cache):
            self.model_names = list()  # no reference caching
        elif model_name in ENSEMBLE_MODELS:
            self.model_names = MODEL_METADATA[model_name]
        else:
            self.model_names = [model_name]

        for model_name in self.model_names:
            assert model_name in self.result.anndata.layers.keys(), (
                f"Model {model_name} not found in the metadata annotation. "
                "You may not using the correct metadata file for this model."
            )

    @staticmethod
    def overlap_genes(
        df_variants,
        df_genes,
        gene_col=None,
        include_cols=None,
        min_from_end=0,
        distance_type="tss",
        min_distance=0,
        max_distance=float("inf"),
    ):
        """Overlap genes with variants.

        Args:
            df_variants: pandas DataFrame containing variants.
            df_genes: pandas DataFrame containing genes.
            gene_col: Column name for gene names.
            include_cols: List of columns to include in the output.
            min_from_end: Minimum distance from the end of the gene.
            distance_type: Type of distance.
            min_distance: Minimum distance from the TSS.
            max_distance: Maximum distance from the TSS.

        Returns:
            pandas DataFrame containing the overlap between genes and variants.

        Examples:
            >>> df_variants = pd.DataFrame(
            ...     {
            ...         "chrom": [
            ...             "1",
            ...             "1",
            ...             "1",
            ...         ],
            ...         "pos": [
            ...             10000,
            ...             10000,
            ...             10000,
            ...         ],
            ...         "ref": [
            ...             "A",
            ...             "A",
            ...             "A",
            ...         ],
            ...         "alt": [
            ...             "G",
            ...             "G",
            ...             "G",
            ...         ],
            ...         "gene": [
            ...             "SPI1",
            ...             "SPI2",
            ...             "SPI3",
            ...         ],
            ...     }
            ... )
            >>> df_genes = pd.DataFrame(
            ...     {
            ...         "gene": [
            ...             "SPI1",
            ...             "SPI2",
            ...             "SPI3",
            ...         ],
            ...         "start": [
            ...             10000,
            ...             10000,
            ...             10000,
            ...         ],
            ...         "end": [
            ...             10000,
            ...             10000,
            ...             10000,
            ...         ],
            ...         "strand": [
            ...             "+",
            ...             "+",
            ...             "+",
            ...         ],
            ...         "gene_mask_start": [
            ...             0,
            ...             0,
            ...             0,
            ...         ],
            ...         "gene_mask_end": [
            ...             4,
            ...             4,
            ...             4,
            ...         ],
            ...     }
            ... )

            >>> df = VariantDataset.overlap_genes(
            ...     df_variants,
            ...     df_genes,
            ... )
            >>> print(df)
            ...    chrom  pos ref alt gene  start end strand  gene_mask_start  gene_mask_end  rel_pos ref_tx alt_tx  tss_dist
            ... 0     1  10000   A   G  SPI1  10000 10000      +              0              4      0      A     G        0
            ... 1     1  10000   A   G  SPI2  10000 10000      +              0              4      0      A     G        0
            ... 2     1  10000   A   G  SPI3  10000 10000      +              0              4      0      A     G        0
        """
        assert min_distance < max_distance, "`min_distance` must be less than `max_distance`"
        include_cols = include_cols or list()

        df_variants = df_variants.copy().astype({"chrom": str})
        if not df_variants["chrom"].str.startswith("chr").any():
            warnings.warn("Chromosome names do not have 'chr' prefix. Adding it to the chromosome names.")
            df_variants["chrom"] = "chr" + df_variants["chrom"].astype(str)
        df_variants["start"] = df_variants.pos.astype(int)
        df_variants["end"] = df_variants["start"] + 1

        if gene_col is not None:
            assert gene_col in df_variants.columns, f"Gene column {gene_col} not found in variants"

            missing = df_variants[~df_variants[gene_col].isin(set(df_genes["gene"]))]
            if len(missing) > 0:
                raise ValueError(
                    f"GeneNotFoundError: Some genes in {gene_col} are not in the result: {missing[gene_col].unique()}"
                )

            df_variants = df_variants.rename(columns={gene_col: "gene"})
            df = df_variants.merge(df_genes, how="left", on="gene", suffixes=("", "_gene"))
        else:
            if "gene" in df_variants.columns:
                warnings.warn(
                    "Gene column `gene` found in variant file."
                    " Overwriting with `gene` column with genes based on the overlap based on genomic coordinates."
                )
                del df_variants["gene"]  # remove gene column from df_genes to avoid duplicate column names
            df = bioframe.overlap(df_genes, df_variants, how="inner", suffixes=("_gene", ""))

        if df.shape[0] == 0:
            raise ValueError(
                "NoOverlapError: There is no overlap between provided variants and genes. Check the provided genes and variants."
            )

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
        df["rel_pos"] = df.apply(VariantDataset._relative_pos, axis=1)
        df["ref_tx"] = df.apply(
            lambda row: row.ref if row.strand == "+" else reverse_complement(row.ref),
            axis=1,
        )
        df["alt_tx"] = df.apply(
            lambda row: row.alt if row.strand == "+" else reverse_complement(row.alt),
            axis=1,
        )
        df["tss_dist"] = df.rel_pos - df.gene_mask_start

        df = df[(df.rel_pos > min_from_end) & (df.rel_pos < DECIMA_CONTEXT_SIZE - min_from_end)]

        if distance_type == "tss":
            df = df[(df.tss_dist.abs() >= min_distance) & (df.tss_dist.abs() < max_distance)]
        elif distance_type == "gene":
            raise NotImplementedError("max_distance_type == 'gene' is not implemented yet.")
        else:
            raise ValueError(f"Invalid distance_type: {distance_type}. Must be 'tss'.")

        if df.shape[0] == 0:
            raise ValueError(
                "NoOverlapError: There is no overlap between provided variants and genes."
                " Check `max_distance` and `min_distance` arguments."
            )

        return df[[*VariantDataset.DEFAULT_COLUMNS, *include_cols]]

    @staticmethod
    def _relative_pos(row):
        # + 1 because gene start is 0 based, variant pos is 1 based and gene end is 1 based
        # TO CONSIDER: relative position is not valid for indels because of the shifting of the sequence.
        # so relative position of reference allele and alterantive allele is different
        return row.pos - (row.start + 1) if row.strand == "+" else row.end - row.pos

    def _overlap_genes(
        self,
        variants,
        include_cols=None,
        gene_col=None,
        min_from_end=0,
        distance_type="tss",
        min_distance=0,
        max_distance=float("inf"),
    ):
        return self.overlap_genes(
            variants,
            self.result.gene_metadata.reset_index(names=["gene"]),
            include_cols=include_cols,
            gene_col=gene_col,
            min_from_end=min_from_end,
            distance_type=distance_type,
            min_distance=min_distance,
            max_distance=max_distance,
        )

    def __len__(self):
        return self.n_seqs * self.n_augmented * self.n_alleles

    def validate_allele_seq(self, gene, variant):
        seq = self.result.gene_sequence(gene, genome=self.genome)
        pos = variant.rel_pos
        ref_match = seq[pos : pos + len(variant.ref)] == variant.ref_tx
        alt_match = seq[pos : pos + len(variant.alt)] == variant.alt_tx
        return ref_match, alt_match

    def predicted_expression_cache(self, gene):
        """Get predicted expression for a gene.

        Args:
            gene: Gene name.

        Returns:
            dict: Dictionary of predicted expression for each model.
        """
        return {model_name: self.result.predicted_gene_expression(gene, model_name) for model_name in self.model_names}

    def __getitem__(self, idx):
        seq_idx, augment_idx, allele_idx = _split_overall_idx(idx, (self.n_seqs, self.n_augmented, self.n_alleles))

        variant = self.variants.iloc[seq_idx]
        rel_pos = variant.rel_pos + self.max_seq_shift

        # by default cache values are nan if matched with reference genome
        # then it will be replaced with the predicted expression from cache.
        pred_expr = {model_name: torch.full((self.result.shape[0],), torch.nan) for model_name in self.model_names}
        ref_match, alt_match = self.validate_allele_seq(variant.gene, variant)

        warnings = list()
        if allele_idx:
            seq, mask = self.result.prepare_one_hot(
                variant.gene,
                variants=[{"chrom": variant.chrom, "pos": variant.pos, "ref": variant.ref, "alt": variant.alt}],
                padding=self.max_seq_shift,
                genome=self.genome,
            )
            allele = seq[:, rel_pos : rel_pos + len(variant.alt)]
            allele_tx = variant.alt_tx

            if alt_match:
                pred_expr = self.predicted_expression_cache(variant.gene)
        else:
            if (not ref_match) and (not alt_match):
                warnings.append(WarningType.ALLELE_MISMATCH_WITH_REFERENCE_GENOME)

            seq, mask = self.result.prepare_one_hot(
                variant.gene,
                variants=[{"chrom": variant.chrom, "pos": variant.pos, "ref": variant.alt, "alt": variant.ref}],
                padding=self.max_seq_shift,
                genome=self.genome,
            )
            allele = seq[:, rel_pos : rel_pos + len(variant.ref)]
            allele_tx = variant.ref_tx

            if ref_match:
                pred_expr = self.predicted_expression_cache(variant.gene)

        if len(variant.ref) == len(variant.alt):  # not SNV there would be shifts
            assert indices_to_strings(allele.argmax(axis=0)) == allele_tx

        inputs = torch.vstack([seq, mask])
        inputs = _extract_center(inputs, seq_len=self.padded_seq_len)
        inputs = self.augmenter(seq=inputs, idx=augment_idx)

        data = {
            "seq": inputs,
            "warning": warnings,
        }
        if len(self.model_names) > 0:
            data["pred_expr"] = pred_expr

        return data

    def collate_fn(self, batch):
        _batch = {
            "seq": default_collate([i["seq"] for i in batch]),
            "warning": list(flatten([b["warning"] for b in batch])),
        }
        if "pred_expr" in batch[0]:
            _batch["pred_expr"] = default_collate([b["pred_expr"] for b in batch])
        return _batch

    def __str__(self):
        return (
            "VariantDataset("
            f"{self.variants.shape[0]} variants "
            f"from {list(self.variants.chrom.unique())} "
            f"between {self.variants.start.min()} "
            f"and {self.variants.end.max()} bp from TSS"
            ")"
        )

    def __repr__(self):
        return self.__str__()
