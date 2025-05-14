from typing import List, Optional, Union
from dataclasses import dataclass
import torch
import anndata
import pandas as pd
import numpy as np
from grelu.visualize import plot_attributions

from decima.data.preprocess import make_inputs
from decima.model.lightning import LightningModel
from decima.plot.visualize import plot_attribution_peaks
from decima.tools.interpret import attributions, find_attr_peaks, scan_attributions


@dataclass
class GeneMetadata:
    """Metadata for a gene in the dataset.
    
    Attributes:
        name: Gene name
        chrom: Chromosome where the gene is located
        start: Start position in the chromosome
        end: End position in the chromosome
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

    @classmethod
    def from_series(cls, name: str, series: pd.Series) -> 'GeneMetadata':
        """Create GeneMetadata from a pandas Series."""
        data = series.to_dict()
        data['name'] = name
        data['fold'] = [f.strip() for f in data['fold'].strip('[]').replace("'", "").split(',')]
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
    region: str
    subregion: str
    celltype_coarse: Optional[str]
    n_cells: int
    total_counts: float
    n_genes: int
    size_factor: float
    train_pearson: float
    val_pearson: float
    test_pearson: float

    @classmethod
    def from_series(cls, name: str, series: pd.Series) -> 'CellMetadata':
        """Create CellMetadata from a pandas Series."""
        data = series.to_dict()
        data['name'] = name
        return cls(**data)


class Attribution:
    """Container for attribution analysis results."""
    
    def __init__(self, gene: str, inputs: torch.Tensor, attrs: np.ndarray, tss_pos: int, n_peaks: int = 10, min_dist: int = 6):
        """Initialize Attribution.
        
        Args:
            inputs: One-hot encoded sequence
            preds: Model predictions
            attrs: Attribution scores
            tss_pos: Transcription start site position
            n_peaks: Number of peaks to find
            min_dist: Minimum distance between peaks
        """
        self.gene = gene
        self.inputs = inputs
        self.attrs = attrs
        self.tss_pos = tss_pos
        self.peaks = self._find_peaks(n_peaks, min_dist)

    def _find_peaks(self, n_peaks: int = 10, min_dist: int = 6):
        # TODO: move to decima.tools.interpret to here but move attribution to interpret
        return find_attr_peaks(self.attrs, tss_pos=self.tss_pos, n=n_peaks, min_dist=min_dist)

    def plot_peaks(self):
        """Plot attribution peaks."""
        return plot_attribution_peaks(self.attrs, self.tss_pos)

    def scan_motifs(self, motifs: str = 'hocomoco_v12', window: int = 18, pthresh: float = 5e-4) -> pd.DataFrame:
        """Scan for motifs in peak regions.
        
        Args:
            motifs: Motif database to use
            n_peaks: Number of top peaks to scan
            window: Window size around peaks
            pthresh: P-value threshold for motif matches
            
        Returns:
            pd.DataFrame: Motif scan results
        """
        return scan_attributions(
            seq=self.inputs[:4], 
            attr=self.attrs, 
            motifs=motifs,
            peaks=self.peaks,
            window=window,
            pthresh=pthresh
        )

    def plot_attributions(self, relative_loc=0, window=50, figsize=(10, 2)):
        """Plot attribution scores around a relative location.
        
        Args:
            relative_loc: Position relative to TSS to center plot on
            window: Number of bases to show on each side of center
            
        Returns:
            matplotlib.pyplot.Figure: Attribution plot
        """
        loc = self.tss_pos + relative_loc
        return plot_attributions(
            self.attrs[:, loc - window:loc + window],
            figsize=figsize)

    def __repr__(self):
        return f"Attribution(gene={self.gene})"

    def __str__(self):
        return f"Attribution(gene={self.gene})"


class DecimaResult:

    def __init__(self, anndata):
        # TODO: allow creating anndata from scratch for given genes and cells and predictions
        self.anndata: anndata.AnnData = anndata
        self._model = None
    
    @classmethod
    def load(cls, anndata_path):
        return cls(anndata.read_h5ad(anndata_path))

    @property
    def model(self):
        if self._model is None:
            self.load_model()
        return self._model

    def load_model(self, model_path: Optional[str] = None, device: str = "cpu"):
        """Load the trained model from a checkpoint path.
        
        Args:
            model_path: Path to model checkpoint. If None, uses self.model_path
        """
        if model_path is None:
            raise NotImplementedError("Default model is not implemented yet.")
        path = model_path # TODO: or use model from wandb
        self._model = LightningModel.load_from_checkpoint(path, device=device)
        self._model.eval()
        return self

    @property
    def genes(self, reliable=None) -> List[str]:
        """List of gene names in the dataset."""
        if reliable is not None:
            raise NotImplementedError("Not implemented and with low pearson correlation should be not reliable.")
        return self.anndata.var_names

    @property
    def cells(self) -> List[str]:
        """List of cell identifiers in the dataset."""
        return self.anndata.obs_names

    @property
    def cell_metadata(self) -> pd.DataFrame:
        """Cell metadata including annotations, metrics, etc."""
        return self.anndata.obs

    @property
    def gene_metadata(self) -> pd.DataFrame:
        """Gene metadata."""
        return self.anndata.var

    def get_gene_metadata(self, gene: str) -> GeneMetadata:
        """Get metadata for a specific gene."""
        if gene not in self.genes:
            raise KeyError(f"Gene {gene} not found in dataset. See avaliable genes with `result.genes`.")
        return GeneMetadata.from_series(gene, self.gene_metadata.loc[gene])

    def get_cell_metadata(self, cell: str) -> CellMetadata:
        """Get metadata for a specific cell."""
        if cell not in self.cells:
            raise KeyError(f"Cell {cell} not found in dataset. See avaliable cells with `result.cells`.")
        return CellMetadata.from_series(cell, self.cell_metadata.loc[cell])

    def predicted_expression_matrix(self, genes: Optional[List[str]] = None) -> pd.DataFrame:
        """Get predicted expression matrix for all or specific genes.
        
        Args:
            genes: Optional list of genes to get predictions for. If None, returns all genes.
            
        Returns:
            pd.DataFrame: Predicted expression matrix (cells x genes)
        """
        if genes is None:
            return pd.DataFrame(self.anndata.layers['preds'], index=self.cells, columns=self.genes)
        else:
            return pd.DataFrame(self.anndata[:, genes].layers['preds'], index=self.cells, columns=genes)

    def prepare_one_hot(self, gene: str) -> torch.Tensor:
        # TODO: move make_inputs to here and deprecate it
        return make_inputs(gene, self.anndata)

    def attributions(
        self, 
        gene: str, 
        cells: Optional[List[str]] = None, 
        constract_cells: Optional[List[str]] = None, 
        transform: str = "specificity",
        n_peaks: int = 10,
        min_dist: int = 6
    ) -> Attribution:
        """Get attributions for a specific gene.

        Args:
            gene: Gene name
            cells: List of cells to use as on task
            constract_cells: List of cells to use as off task
            transform: Attribution transform method
            n_peaks: Number of peaks to find
            min_dist: Minimum distance between peaks
            
        Returns:
            Attribution: Container with inputs, predictions, attribution scores and TSS position
        """
        if isinstance(cells, str):
            cells = self.query_cells(cells)

        if isinstance(constract_cells, str):
            constract_cells = self.query_cells(constract_cells)

        one_hot_seq, gene_mask = self.prepare_one_hot(gene)
        inputs = torch.vstack([one_hot_seq, gene_mask])

        attrs, tss_pos = attributions(
            gene=gene, inputs=inputs, model=self._model, device=self.model.device, 
            tasks=cells,
            off_tasks=constract_cells,
            transform=transform)

        return Attribution(
            gene=gene,
            inputs=inputs,
            attrs=attrs,
            tss_pos=tss_pos,
            n_peaks=n_peaks,
            min_dist=min_dist
        )

    def query_cells(self, query: str):
        return self.cell_metadata.query(query).index.tolist()

    @property
    def shape(self) -> tuple:
        """Shape of the expression matrix (n_cells, n_genes)."""
        return self.anndata.shape

    def __repr__(self):
        return f"DecimaResult(anndata={self.anndata})"

    def __str__(self):
        return f"DecimaResult(anndata={self.anndata})"