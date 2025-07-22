from typing import Dict, List, Optional, Union
import anndata
import numpy as np
import torch
import pandas as pd

from grelu.sequence.format import intervals_to_strings, strings_to_one_hot

from decima.constants import DECIMA_CONTEXT_SIZE
from decima.hub import load_decima_metadata, load_decima_model
from decima.core.metadata import GeneMetadata, CellMetadata
from decima.utils.inject import prepare_seq_alt_allele


class DecimaResult:
    """
    Container for Decima results and model predictions.

    This class provides a unified interface for loading pre-trained Decima models and
    associated metadata, making predictions, and performing attribution analyses.

    The DecimaResult object contains:
        - An AnnData object with gene expression and metadata
        - A trained model for making predictions
        - Methods for attribution analysis and interpretation

    Args:
        anndata: AnnData object containing gene expression data and metadata

    Examples:
        >>> # Load default pre-trained model and metadata
        >>> result = DecimaResult.load()
        >>> result.load_model(
        ...     rep=0
        ... )
        >>> # Perform attribution analysis
        >>> attributions = result.attributions(
        ...     output_dir="attrs_SP1I_classical_monoctypes",
        ...     gene="SPI1",
        ...     tasks='cell_type == "classical monocyte"',
        ... )

    Properties:
        model: Decima model
        genes: List of gene names
        cells: List of cell names
        cell_metadata: Cell metadata
        gene_metadata: Gene metadata
        shape: Shape of the expression matrix
        attributions: Attributions for a gene
    """

    def __init__(self, anndata):
        self.anndata: anndata.AnnData = anndata
        self._model = None

    @classmethod
    def load(cls, anndata_path: Optional[Union[str, anndata.AnnData]] = None):
        """Load a DecimaResult object from an anndata file or a path to an anndata file.

        Args:
            anndata_path: Path to anndata file or anndata object

        Returns:
            DecimaResult object

        Examples:
            >>> result = DecimaResult.load()  # Load default decima metadata
            >>> result = DecimaResult.load(
            ...     "path/to/anndata.h5ad"
            ... )  # Load custom anndata object from file
        """
        if anndata_path is None:
            return cls(load_decima_metadata())
        elif isinstance(anndata_path, str):
            return cls(anndata.read_h5ad(anndata_path))
        elif isinstance(anndata_path, anndata.AnnData):
            return cls(anndata_path)
        else:
            raise ValueError(f"Invalid anndata path: {anndata_path}")

    @property
    def model(self):
        """Decima model."""
        if self._model is None:
            self.load_model()
        return self._model

    def load_model(self, model: Optional[Union[str, int]] = 0, device: str = "cpu"):
        """Load the trained model from a checkpoint path.

        Args:
            model: Path to model checkpoint or replicate number (0-3) for pre-trained models
            device: Device to load model on

        Returns:
            self

        Examples:
            >>> result = DecimaResult.load()
            >>> result.load_model()  # Load default model (rep0)
            >>> result.load_model(
            ...     model="path/to/checkpoint.ckpt"
            ... )
            >>> result.load_model(
            ...     model=2
            ... )
        """
        self._model = load_decima_model(model=model, device=device)
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

    def predicted_expression_matrix(
        self, genes: Optional[List[str]] = None, model_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Get predicted expression matrix for all or specific genes.

        Args:
            genes: Optional list of genes to get predictions for. If None, returns all genes.

        Returns:
            pd.DataFrame: Predicted expression matrix (cells x genes)
        """
        model_name = "preds" if (model_name is None) or (model_name == "ensemble") else model_name
        if genes is None:
            return pd.DataFrame(self.anndata.layers[model_name], index=self.cells, columns=self.genes)
        else:
            return pd.DataFrame(self.anndata[:, genes].layers[model_name], index=self.cells, columns=genes)

    def predicted_gene_expression(self, gene, model_name):
        return torch.from_numpy(self.anndata[:, gene].layers[model_name].ravel())

    def _pad_gene_metadata(self, gene_meta: pd.Series, padding: int = 0) -> pd.Series:
        """
        Pad gene metadata with padding.

        Args:
            gene_meta: Gene metadata
            padding: Padding to add to the gene metadata

        Returns:
            pd.Series: Padded gene metadata
        """
        gene_meta = gene_meta.copy()
        gene_meta["start"] = gene_meta["start"] - padding
        gene_meta["end"] = gene_meta["end"] + padding
        gene_meta["gene_mask_start"] = gene_meta["gene_mask_start"] + padding
        gene_meta["gene_mask_end"] = gene_meta["gene_mask_end"] + padding
        return gene_meta

    def prepare_one_hot(self, gene: str, variants: Optional[List[Dict]] = None, padding: int = 0) -> torch.Tensor:
        """Prepare one-hot encoding for a gene.

        Args:
            gene: Gene name

        Returns:
            torch.Tensor: One-hot encoding of the gene
        """

        assert gene in self.genes, f"{gene} is not in the anndata object"
        gene_meta = self._pad_gene_metadata(self.gene_metadata.loc[gene], padding)

        if variants is None:
            seq = intervals_to_strings(gene_meta, genome="hg38")
            gene_start, gene_end = gene_meta.gene_mask_start, gene_meta.gene_mask_end
        else:
            seq, (gene_start, gene_end) = prepare_seq_alt_allele(gene_meta, variants)

        mask = np.zeros(shape=(1, DECIMA_CONTEXT_SIZE + padding * 2))
        mask[0, gene_start:gene_end] += 1
        mask = torch.from_numpy(mask).float()

        return strings_to_one_hot(seq), mask

    def gene_sequence(self, gene: str, stranded: bool = True) -> str:
        """Get sequence for a gene."""

        try:
            assert gene in self.genes, f"{gene} is not in the anndata object"
        except AssertionError:
            print(gene)
            print(self.genes)
        gene_meta = self.gene_metadata.loc[gene]
        if not stranded:
            gene_meta = {"chrom": gene_meta.chrom, "start": gene_meta.start, "end": gene_meta.end}
        return intervals_to_strings(gene_meta, genome="hg38")

    def attributions(
        self,
        gene: str,
        tasks: Optional[List[str]] = None,
        off_tasks: Optional[List[str]] = None,
        transform: str = "specificity",
        method: str = "inputxgradient",
        threshold: float = 5e-4,
        min_seqlet_len: int = 4,
        max_seqlet_len: int = 25,
        additional_flanks: int = 0,
    ):
        """Get attributions for a specific gene.

        Args:
            gene: Gene name
            tasks: List of cells to use as on task
            off_tasks: List of cells to use as off task
            transform: Attribution transform method
            method: Attribution method
            n_peaks: Number of peaks to find
            min_dist: Minimum distance between peaks

        Returns:
            Attribution: Container with inputs, predictions, attribution scores and TSS position
        """
        tasks, off_tasks = self.query_tasks(tasks, off_tasks)

        one_hot_seq, gene_mask = self.prepare_one_hot(gene)
        inputs = torch.vstack([one_hot_seq, gene_mask])

        from decima.interpret.attributions import Attribution, attributions  # to avoid circular import

        attrs = attributions(
            inputs=inputs.unsqueeze(0),
            model=self.model,
            tasks=tasks,
            off_tasks=off_tasks,
            transform=transform,
            device=self.model.device,
            method=method,
        ).squeeze(0)

        gene_meta = self.gene_metadata.loc[gene]
        return Attribution(
            gene=gene,
            inputs=inputs,
            attrs=attrs,
            chrom=gene_meta.chrom,
            start=gene_meta.start,
            end=gene_meta.end,
            strand=gene_meta.strand,
            threshold=threshold,
            min_seqlet_len=min_seqlet_len,
            max_seqlet_len=max_seqlet_len,
            additional_flanks=additional_flanks,
        )

    def query_cells(self, query: str):
        return self.cell_metadata.query(query).index.tolist()

    def query_tasks(self, tasks: Optional[List[str]] = None, off_tasks: Optional[List[str]] = None):
        if tasks is None:
            tasks = self.cell_metadata.index.tolist()
        elif isinstance(tasks, str):
            tasks = self.query_cells(tasks)

        if isinstance(off_tasks, str):
            off_tasks = self.query_cells(off_tasks)

        return tasks, off_tasks

    @property
    def shape(self) -> tuple:
        """Shape of the expression matrix (n_cells, n_genes)."""
        return self.anndata.shape

    def __repr__(self):
        return f"DecimaResult(anndata={self.anndata})"

    def __str__(self):
        return f"DecimaResult(anndata={self.anndata})"
