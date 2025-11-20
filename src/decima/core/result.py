from typing import Dict, List, Optional, Union
import anndata
import numpy as np
import torch
import pandas as pd
from scipy import stats

from grelu.sequence.format import intervals_to_strings, strings_to_one_hot

from decima.constants import DECIMA_CONTEXT_SIZE, DEFAULT_ENSEMBLE, MODEL_METADATA
from decima.hub import load_decima_metadata, load_decima_model
from decima.core.metadata import GeneMetadata, CellMetadata
from decima.tools.evaluate import marker_zscores
from decima.plot.visualize import import_plotnine
from decima.utils.inject import prepare_seq_alt_allele
from decima.interpret.attributer import DecimaAttributer  # to avoid circular import


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
    def load(cls, anndata_name_or_path: Optional[Union[str, anndata.AnnData]] = None):
        """Load a DecimaResult object from an anndata file or a path to an anndata file.

        Args:
            anndata_name_or_path: Name of the model or path to anndata file or anndata object
            model: Model name or path to model checkpoint. If not provided, the default model will be loaded.

        Returns:
            DecimaResult object

        Examples:
            >>> result = DecimaResult.load()  # Load default decima metadata
            >>> result = DecimaResult.load(
            ...     "path/to/anndata.h5ad"
            ... )  # Load custom anndata object from file
        """
        if isinstance(anndata_name_or_path, list):
            anndata_name_or_path = anndata_name_or_path[0]

        if (anndata_name_or_path is None) or (anndata_name_or_path in MODEL_METADATA):
            return cls(load_decima_metadata(name_or_path=anndata_name_or_path))
        elif isinstance(anndata_name_or_path, str):
            return cls(anndata.read_h5ad(anndata_name_or_path))
        elif isinstance(anndata_name_or_path, anndata.AnnData):
            return cls(anndata_name_or_path)
        elif isinstance(anndata_name_or_path, DecimaResult):
            return anndata_name_or_path
        else:
            raise ValueError(f"Invalid anndata path: {anndata_name_or_path}")

    @property
    def model(self):
        """Decima model."""
        if self._model is None:
            self.load_model()
        return self._model

    def load_model(self, model: Optional[Union[str, int]] = MODEL_METADATA[DEFAULT_ENSEMBLE][0], device: str = "cpu"):
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

    def assert_genes(self, genes: List[str]) -> bool:
        """Check if the genes are in the dataset."""
        missing_genes = set(genes) - set(self.genes)
        if missing_genes:
            raise ValueError(f"Genes {missing_genes} are not in the dataset. See avaliable genes with `result.genes`.")

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

    @property
    def ground_truth(self) -> pd.DataFrame:
        """Ground truth expression matrix."""
        return self.anndata.X

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
        model_name = "preds" if (model_name is None) or (model_name in MODEL_METADATA) else model_name
        if genes is None:
            return pd.DataFrame(self.anndata.layers[model_name], index=self.cells, columns=self.genes)
        else:
            return pd.DataFrame(self.anndata[:, genes].layers[model_name], index=self.cells, columns=genes)

    def predicted_gene_expression(self, gene, model_name):
        """Get predicted expression for a specific gene.

        Args:
            gene: Gene name
            model_name: Model name

        Returns:
            torch.Tensor: Predicted expression for the gene
        """
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

    def prepare_one_hot(
        self, gene: str, variants: Optional[List[Dict]] = None, padding: int = 0, genome: str = "hg38"
    ) -> torch.Tensor:
        """Prepare one-hot encoding for a gene.

        Args:
            gene: Gene name
            variants: Optional list of variant dictionaries to inject into the sequence
            padding: Amount of padding to add on both sides of the sequence
            genome: Genome name or path to the genome fasta file. Default: "hg38"

        Returns:
            torch.Tensor: One-hot encoding of the gene
        """
        assert gene in self.genes, f"{gene} is not in the anndata object. See avaliable genes with `result.genes`."
        gene_meta = self._pad_gene_metadata(self.gene_metadata.loc[gene], padding)

        if variants is None:
            seq = intervals_to_strings(gene_meta, genome=genome)
            gene_start, gene_end = gene_meta.gene_mask_start, gene_meta.gene_mask_end
        else:
            # Todo: fix for case where genome is not hg38
            seq, (gene_start, gene_end) = prepare_seq_alt_allele(gene_meta, variants, genome=genome)

        mask = np.zeros(shape=(1, DECIMA_CONTEXT_SIZE + padding * 2))
        mask[0, gene_start:gene_end] += 1
        mask = torch.from_numpy(mask).float()

        return strings_to_one_hot(seq), mask

    def gene_sequence(self, gene: str, stranded: bool = True, genome: str = "hg38") -> str:
        """Get sequence for a gene.

        Args:
            gene: Gene name
            stranded: Whether to return stranded sequence
            genome: Genome name or path to the genome fasta file. Default: "hg38"

        Returns:
            str: Sequence for the gene
        """
        assert gene in self.genes, f"{gene} is not in the anndata object. See avaliable genes with `result.genes`."
        gene_meta = self.gene_metadata.loc[gene]
        if not stranded:
            gene_meta = {"chrom": gene_meta.chrom, "start": gene_meta.start, "end": gene_meta.end}
        return intervals_to_strings(gene_meta, genome=genome)

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
        genome: str = "hg38",
    ):
        """Get attributions for a specific gene.

        Args:
            gene: Gene name
            tasks: List of cells to use as on task
            off_tasks: List of cells to use as off task
            transform: Attribution transform method
            method: Method to use for attribution analysis available options: "saliency", "inputxgradient", "integratedgradients".
            threshold: Threshold for attribution analysis
            min_seqlet_len: Minimum length for seqlet calling
            max_seqlet_len: Maximum length for seqlet calling
            additional_flanks: Additional flanks for seqlet calling
            genome: Genome to use for attribution analysis default is "hg38". Can be genome name or path to custom genome fasta file.

        Returns:
            Attribution: Container with inputs, predictions, attribution scores and TSS position
        """
        tasks, off_tasks = self.query_tasks(tasks, off_tasks)

        one_hot_seq, gene_mask = self.prepare_one_hot(gene, genome=genome)
        inputs = torch.vstack([one_hot_seq, gene_mask])

        attrs = (
            DecimaAttributer(
                model=self.model,
                tasks=tasks,
                off_tasks=off_tasks,
                transform=transform,
                method=method,
            )
            .attribute(
                inputs=inputs.unsqueeze(0),
            )
            .squeeze(0)
            .cpu()
            .numpy()
        )

        gene_meta = self.gene_metadata.loc[gene]
        from decima.core.attribution import Attribution  # to avoid circular import

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
        """Query cells based on a query string.

        Args:
            query: Query string

        Returns:
            List of cell names

        Examples:
            >>> result = DecimaResult.load()
            >>> cells = result.query_cells(
            ...     "cell_type == 'classical monocyte'"
            ... )
            >>> cells
            ['agg1', 'agg2', 'agg3', ...]
        """
        return self.cell_metadata.query(query).index.tolist()

    def query_tasks(self, tasks: Optional[List[str]] = None, off_tasks: Optional[List[str]] = None):
        """Query tasks based on a query string.

        Args:
            tasks: Query string
            off_tasks: Query string

        Returns:
            List of tasks

        Examples:
            >>> result = DecimaResult.load()
            >>> tasks = result.query_tasks(
            ...     "cell_type == 'classical monocyte'"
            ... )
            >>> tasks
            [...]
        """
        if tasks is None:
            tasks = self.cell_metadata.index.tolist()
        elif isinstance(tasks, str):
            tasks = self.query_cells(tasks)

        if isinstance(off_tasks, str):
            off_tasks = self.query_cells(off_tasks)

        return tasks, off_tasks

    def marker_zscores(self, tasks, off_tasks=None, layer="preds"):
        """Compute marker z-scores to identify differentially expressed genes.

        Args:
            tasks: Target cells. Query string or list of cell IDs.
            off_tasks: Background cells. Query string, list of cell IDs, or None
                (uses all other cells).
            layer: Expression data layer. "preds" (default), "expression", or
                custom layer name.

        Returns:
            pandas.DataFrame: Columns are 'gene', 'score' (z-score), 'task'.

        Examples:
            >>> # Classical monocytes vs all others
            >>> markers = result.marker_zscores(
            ...     "cell_type == 'classical monocyte'"
            ... )
            >>> top_genes = markers.nlargest(
            ...     10, "score"
            ... )

            >>> markers = result.marker_zscores(
            ...     tasks="cell_type == 'classical monocyte'",
            ...     off_tasks="cell_type == 'lymphoid progenitor'",
            ... )
        """
        off_tasks = off_tasks or []

        if layer == "expression":
            layer = None  # use ground truth expression

        tasks, off_tasks = self.query_tasks(tasks, off_tasks)
        all_tasks = set(tasks).union(set(off_tasks))

        ad = self.anndata[self.anndata.obs.index.isin(all_tasks)].copy()
        ad.obs.loc[off_tasks, "task"] = "off"
        ad.obs.loc[tasks, "task"] = "on"

        return marker_zscores(ad, key="task", layer=layer).sort_values(by="score", ascending=False)

    def _correlation(self, tasks, off_tasks=None, dataset="test"):
        if self.ground_truth is None:
            raise ValueError(
                "Anndata object does not have expression data. Cannot compute correlations between tasks and off tasks."
            )

        tasks, off_tasks = self.query_tasks(tasks, off_tasks)
        tasks = self.anndata.obs.index.isin(tasks)

        preds = self.anndata.layers["preds"][tasks].mean(axis=0)
        true = self.anndata.X[tasks].mean(axis=0)

        if off_tasks is not None:
            off_tasks = self.anndata.obs.index.isin(off_tasks)
            preds = preds - self.anndata.layers["preds"][off_tasks].mean(axis=0)
            true = true - self.anndata.X[off_tasks].mean(axis=0)

        if dataset is not None:
            genes = self.anndata.var.dataset == dataset
            preds = preds[genes]
            true = true[genes]

        return true, preds

    def correlation(self, tasks, off_tasks, dataset="test"):
        """Compute the correlation between the ground truth and the predicted expression.

        Args:
            tasks: List of cells to use as on task.
            off_tasks: List of cells to use as off task.
            dataset: Dataset to use for computation.

        Returns:
            float: Pearson correlation coefficient.
        """
        ground_truth, preds = self._correlation(tasks, off_tasks, dataset)
        return stats.pearsonr(ground_truth, preds)

    def plot_correlation(self, tasks, off_tasks, dataset="test"):
        """Plot the correlation between the ground truth and the predicted expression.

        Args:
            tasks: List of cells to use as on task.
            off_tasks: List of cells to use as off task.
            dataset: Dataset to use for computation.

        Returns:
            p9.ggplot: Plot of the correlation between the ground truth and the predicted expression.

        Examples:
            >>> result = DecimaResult.load()
            >>> result.plot_correlation(
            ...     tasks="cell_type == 'classical monocyte'",
            ...     off_tasks="cell_type == 'lymphoid progenitor'",
            ... )
        """
        p9 = import_plotnine()
        true, preds = self._correlation(tasks, off_tasks, dataset)
        pearsonr = stats.pearsonr(true, preds)
        df = pd.DataFrame(
            {
                "true": true,
                "pred": preds,
            }
        )
        return (
            p9.ggplot(df, p9.aes(x="true", y="pred"))
            + p9.geom_pointdensity(size=0.1)
            + p9.xlab("Measured log FC")
            + p9.ylab("Predicted logFC")
            + p9.geom_text(
                x=df["true"].min() * 0.5,
                y=df["pred"].max() * 0.95,
                label=f"rho={pearsonr[0]:.2f} (P={pearsonr[1]:.2e})",
            )
            + p9.geom_abline(slope=1, intercept=0)
            + p9.geom_vline(xintercept=0, linetype="--")
            + p9.geom_hline(yintercept=0, linetype="--")
        )

    @property
    def shape(self) -> tuple:
        """Shape of the expression matrix (n_cells, n_genes)."""
        return self.anndata.shape

    def __repr__(self):
        return f"DecimaResult(anndata={self.anndata})"

    def __str__(self):
        return f"DecimaResult(anndata={self.anndata})"
