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
from decima.core.metadata import GeneMetadata, CellMetadata
from decima.interpret.attribution import Attribution, attributions
from decima.interpret.ism import ism # TODO: implement ism



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

        attrs = attributions(
            gene=gene, inputs=inputs, model=self._model, device=self.model.device, 
            tasks=cells,
            off_tasks=constract_cells,
            transform=transform)

        gene_meta = self.gene_metadata.loc[gene]
        return Attribution(
            gene=gene,
            inputs=inputs,
            attrs=attrs,
            chrom=gene_meta.chrom,
            start=gene_meta.start,
            end=gene_meta.end,
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