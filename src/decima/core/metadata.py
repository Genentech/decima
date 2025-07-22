from typing import List, Optional
from dataclasses import dataclass
import pandas as pd


@dataclass
class GeneMetadata:
    """Metadata for a gene in the dataset.

    Attributes:
        name: Gene name
        chrom: Chromosome where the gene is located
        start: Start position of the region around the gene to perform predictions in the chromosome
        end: End position of the region around the gene to perform predictions in the chromosome
        strand: Strand orientation (+ or -)
        gene_type: Type of gene (e.g., protein_coding)
        frac_nan: Fraction of NaN values
        mean_counts: Mean count across samples
        n_tracks: Number of tracks
        gene_start: Gene start position
        gene_end: Gene end position
        gene_length: Length of the gene
        gene_mask_start: Start position of the gene mask
        gene_mask_end: End position of the gene mask
        frac_N: Fraction of N bases
        fold: Cross-validation fold
        dataset: Dataset identifier
        gene_id: Ensembl gene ID
        pearson: Pearson correlation
        size_factor_pearson: Size factor Pearson correlation
    """

    name: str
    chrom: str
    start: int
    end: int
    strand: str
    gene_type: str
    frac_nan: float
    mean_counts: float
    n_tracks: int
    gene_start: int
    gene_end: int
    gene_length: int
    gene_mask_start: int
    gene_mask_end: int
    frac_N: float
    fold: List[str]
    dataset: str
    gene_id: str
    pearson: float
    size_factor_pearson: float
    ensembl_canonical_tss: Optional[bool]

    @classmethod
    def from_series(cls, name: str, series: pd.Series) -> "GeneMetadata":
        """Create GeneMetadata from a pandas Series."""
        data = series.to_dict()
        data["name"] = name
        data["fold"] = [f.strip() for f in data["fold"].strip("[]").replace("'", "").split(",")]
        return cls(**data)


@dataclass
class CellMetadata:
    """Metadata for a cell in the dataset.

    Attributes:
        name: Cell identifier
        cell_type: Detailed cell type
        tissue: Tissue identifier
        organ: Organ name
        disease: Disease state
        study: Study identifier
        dataset: Dataset identifier
        region: Anatomical region
        subregion: Anatomical subregion
        celltype_coarse: Coarse cell type classification
        n_cells: Number of cells
        total_counts: Total count of transcripts
        n_genes: Number of genes detected
        size_factor: Size normalization factor
        train_pearson: Pearson correlation in training set
        val_pearson: Pearson correlation in validation set
        test_pearson: Pearson correlation in test set
    """

    name: str
    cell_type: str
    tissue: str
    organ: str
    disease: str
    study: str
    dataset: str
    region: Optional[str]
    subregion: Optional[str]
    celltype_coarse: Optional[str]
    n_cells: int
    total_counts: float
    n_genes: int
    size_factor: float
    train_pearson: float
    val_pearson: float
    test_pearson: float

    @classmethod
    def from_series(cls, name: str, series: pd.Series) -> "CellMetadata":
        """Create CellMetadata from a pandas Series."""
        data = series.to_dict()
        data["name"] = name
        return cls(**data)
