from typing import Iterator, Optional
from pathlib import Path

import h5py
import torch
import numpy as np
import pandas as pd
import pyBigWig
import genomepy
from pyfaidx import Fasta
from grelu.sequence.format import convert_input_type

from decima.constants import DECIMA_CONTEXT_SIZE
from decima.core.result import DecimaResult


def import_cyvcf2():
    try:
        from cyvcf2 import VCF  # optional dependency
    except ImportError:
        raise ImportError(
            "Optional dependency `cyvcf2` is not installed and is required for reading vcf files. "
            "Please install vep dependencies with `pip install decima[vep]`."
        )
    return VCF


def read_fasta_gene_mask(fasta_file: str) -> pd.DataFrame:
    """Read the fasta file and return the gene mask

    Args:
        fasta_file (str): Path to the fasta file

    Returns:
        pd.DataFrame: DataFrame with the gene mask
    """
    df = list()
    with Fasta(fasta_file) as fasta:
        for record in fasta:
            name, start, end = record.name.split("|")
            start_label, start = start.split("=")
            end_label, end = end.split("=")
            assert (start_label == "gene_mask_start") and (end_label == "gene_mask_end"), (
                "First boundary label must be `gene_mask_start` and second boundary label must be `gene_mask_end` "
                "for example: `>seq|gene_mask_start=1|gene_mask_end=10`."
            )
            seq = str(record[:])
            assert (
                len(seq) == DECIMA_CONTEXT_SIZE
            ), f"Sequence length must be equal to {DECIMA_CONTEXT_SIZE}. Found {len(seq)} in the fasta file: {fasta_file}"
            df.append({"gene": name, "seq": seq, "gene_mask_start": int(start), "gene_mask_end": int(end)})
        return pd.DataFrame(df).set_index("gene")


def read_vcf_chunks(vcf_file: str, chunksize: int) -> Iterator[pd.DataFrame]:
    """Read the vcf file and return the chunks

    Args:
        vcf_file (str): Path to the vcf file
        chunksize (int): Size of the chunks

    Returns:
        Iterator[pd.DataFrame]: Iterator of DataFrames with the chunks
    """
    VCF = import_cyvcf2()
    vcf = VCF(vcf_file)
    df = list()

    for record in vcf:
        alt_allele = record.ALT if record.ALT else [""]
        for alt in alt_allele:
            df.append(
                {
                    "chrom": record.CHROM,
                    "pos": record.POS,
                    "ref": record.REF,
                    "alt": alt,
                }
            )
        if len(df) >= chunksize:
            yield pd.DataFrame(df)
            df = list()

    if df:
        yield pd.DataFrame(df)

    vcf.close()


class BigWigWriter:
    """Write genomic data to BigWig format for genome browser visualization.

    Opens a BigWig file for writing, accumulates data, then writes and closes on exit.

    Args:
        path: Output BigWig file path.
        genome: Reference genome name for chromosome sizes.

    Examples:
        >>> with BigWigWriter(
        ...     "output.bigwig",
        ...     "hg38",
        ... ) as writer:
        ...     writer.add(
        ...         "chr1",
        ...         1000,
        ...         2000,
        ...         attribution_scores,
        ...     )
    """

    def __init__(self, path, genome: str = "hg38", threshold: float = 1e-5):
        self.path = path
        self.genome = genome
        self.threshold = threshold
        if isinstance(genome, str):
            self.sizes = genomepy.Genome(genome).sizes
        elif isinstance(genome, list):
            self.sizes = {g: DECIMA_CONTEXT_SIZE for g in genome}
        else:
            raise ValueError(
                f"Invalid type for genome: {type(genome)}. Either provide genome name like `hg38` or list of gene names."
            )
        self.measures = dict()

    def open(self):
        """Open BigWig file for writing and add chromosome header."""
        self.bw = pyBigWig.open(self.path, "w")
        self.bw.addHeader([(chrom, size) for chrom, size in self.sizes.items()])

    def __enter__(self):
        """Context manager entry - opens BigWig file."""
        self.open()
        return self

    def add(self, chrom: str, start: int, end: int, values: np.ndarray):
        """Add genomic interval data to be written.

        Args:
            chrom: Chromosome name.
            start: Start position.
            end: End position.
            values: Array of values for each position.
        """
        if chrom not in self.measures:
            self.measures[chrom] = {
                "values": np.zeros(self.sizes[chrom]),
                "count": np.zeros(self.sizes[chrom], dtype=int),
            }
        self.measures[chrom]["values"][start:end] += values
        self.measures[chrom]["count"][start:end] += 1

    def close(self):
        """Write accumulated data to BigWig file and close."""
        for chrom in self.sizes.keys():
            if chrom not in self.measures:
                continue
            data = self.measures[chrom]
            pos = np.where(np.abs(data["values"]) > self.threshold)[0]
            if len(pos) == 0:
                continue
            values = data["values"][pos] / data["count"][pos]
            self.bw.addEntries(chrom, pos, values=values, span=1, step=1)
        self.bw.close()

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit - closes BigWig file."""
        self.close()


class AttributionWriter:
    """Write gene attribution data to HDF5 and BigWig files.

    Output files:
        HDF5 file:
            - genes: Gene names (string array)
            - attribution: Attribution scores (genes x 4 x context_size)
            - sequence: One-hot sequences (genes x 4 x context_size)
            - Attributes: model_name, genome
        BigWig file: Mean attribution scores at genomic coordinates

    Args:
        path: Output HDF5 file path.
        genes: Gene names to write.
        model_name: Model identifier for metadata.
        metadata_anndata: Gene metadata source or path to the metadata anndata file. If not provided, the compatible metadata for the model will be used.
        genome: Reference genome version.
        bigwig: Create BigWig file for genome browser.
        correct_grad_bigwig: Correct gradient bigwig for bias.
        custom_genes: If True, do not assert that the genes are in the result.

    Examples:
        >>> with (
        ...     AttributionWriter(
        ...         "attrs.h5",
        ...         ["SPI1"],
        ...         "model_v1",
        ...     ) as writer
        ... ):
        ...     writer.add(
        ...         "SPI1",
        ...         attribution_array,
        ...         sequence_array,
        ...     )
    """

    def __init__(
        self,
        path,
        genes,
        model_name,
        metadata_anndata=None,
        genome: str = "hg38",
        bigwig: bool = True,
        correct_grad_bigwig: bool = True,
        custom_genes: bool = False,
    ):
        self.path = path
        self.genes = genes
        self.genome = genome
        self.bigwig = bigwig
        self.model_name = model_name
        self.idx = {g: i for i, g in enumerate(self.genes)}
        self.result = DecimaResult.load(metadata_anndata or model_name)
        self.correct_grad_bigwig = correct_grad_bigwig
        self.custom_genes = custom_genes

    def open(self):
        """Open HDF5 file and optional BigWig file for writing."""
        if not self.custom_genes:
            self.result.assert_genes(self.genes)

        self.h5_writer = h5py.File(self.path, "w")
        self.h5_writer.attrs["model_name"] = self.model_name
        self.h5_writer.attrs["genome"] = self.genome
        self.h5_writer.create_dataset(
            "genes",
            (len(self.genes),),
            dtype="S100",
            compression="gzip",
        )
        self.h5_writer.create_dataset(
            "gene_mask_start",
            (len(self.genes),),
            dtype="i4",
            compression="gzip",
        )
        self.h5_writer.create_dataset(
            "gene_mask_end",
            (len(self.genes),),
            dtype="i4",
            compression="gzip",
        )
        self.h5_writer["genes"][:] = np.array(self.genes, dtype="S100")
        self.h5_writer.create_dataset(
            "attribution",
            (len(self.genes), 4, DECIMA_CONTEXT_SIZE),
            chunks=(1, 4, DECIMA_CONTEXT_SIZE),
            dtype="float32",
            compression="gzip",
        )
        self.h5_writer.create_dataset(
            "sequence",
            (len(self.genes), DECIMA_CONTEXT_SIZE),
            chunks=(1, DECIMA_CONTEXT_SIZE),
            dtype="i1",
            compression="gzip",
        )
        if self.bigwig:
            self.bigwig_writer = BigWigWriter(
                Path(self.path).with_suffix(".bigwig").as_posix(),
                genome=self.genes if self.custom_genes else self.genome,
            )
            self.bigwig_writer.open()

    def __enter__(self):
        """Context manager entry - opens files for writing."""
        self.open()
        return self

    def add(
        self,
        gene: str,
        seqs: np.ndarray,
        attrs: np.ndarray,
        gene_mask_start: Optional[int] = None,
        gene_mask_end: Optional[int] = None,
    ):
        """Add attribution data for a gene.

        Args:
            gene: Gene name from the genes list.
            attrs: Attribution scores, shape (4, DECIMA_CONTEXT_SIZE).
            seqs: One-hot DNA sequence, shape (4, DECIMA_CONTEXT_SIZE).
            gene_mask_start: Gene mask start position. If None, the gene mask start position will be loaded from the result.
            gene_mask_end: Gene mask end position. If None, the gene mask end position will be loaded from the result.
        """
        assert seqs.shape == (
            4,
            DECIMA_CONTEXT_SIZE,
        ), "`seqs` must be 4-dimensional with shape (4, DECIMA_CONTEXT_SIZE)."
        assert attrs.shape == (
            4,
            DECIMA_CONTEXT_SIZE,
        ), "`attrs` must be 4-dimensional with shape (4, DECIMA_CONTEXT_SIZE)."

        if (gene_mask_start is None) and (gene_mask_end is None):
            assert (
                not self.custom_genes
            ), "`gene_mask_start` and `gene_mask_end` must be provided if `custom_genes` is True."
            gene_mask_start = self.result.gene_metadata.loc[gene, "gene_mask_start"].astype("i4")
            gene_mask_end = self.result.gene_metadata.loc[gene, "gene_mask_end"].astype("i4")
        elif (gene_mask_start is not None) and (gene_mask_end is not None):
            pass
        else:
            raise ValueError(
                "Either `gene_mask_start` and `gene_mask_end` must be provided together or both must be None."
            )
        idx = self.idx[gene]
        self.h5_writer["gene_mask_start"][idx] = int(gene_mask_start)
        self.h5_writer["gene_mask_end"][idx] = int(gene_mask_end)
        self.h5_writer["sequence"][idx, :] = convert_input_type(
            torch.from_numpy(seqs),  # convert_input_type only support Tensor
            "indices",
            input_type="one_hot",
        )[np.newaxis].astype("i1")
        self.h5_writer["attribution"][idx, :, :] = attrs.astype("float32")

        if self.bigwig:
            if self.custom_genes:
                chrom = gene
                start = 0
                end = DECIMA_CONTEXT_SIZE
            else:
                gene_meta = self.result.get_gene_metadata(self.genes[idx])
                chrom = gene_meta.chrom
                start = gene_meta.start
                end = gene_meta.end

            if self.correct_grad_bigwig:
                attrs = attrs - attrs.mean(axis=0, keepdims=True)

            self.bigwig_writer.add(chrom, start, end, (attrs * seqs).mean(axis=0))

    def close(self):
        """Close HDF5 file and optional BigWig file."""
        self.h5_writer.close()
        if self.bigwig:
            self.bigwig_writer.close()

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit - closes files."""
        self.close()


class VariantAttributionWriter(AttributionWriter):
    def __init__(
        self,
        path,
        genes,
        variants,
        model_name,
        metadata_anndata=None,
        genome: str = "hg38",
    ):
        super().__init__(
            path,
            genes,
            model_name,
            metadata_anndata,
            genome,
            bigwig=False,
            custom_genes=False,
        )
        assert len(variants) == len(
            genes
        ), "Number of variants must be equal to number of genes. AttributionWriter saves variant gene pairs."
        self.idx = {(v, g): i for i, (v, g) in enumerate(zip(variants, genes))}
        self.variants = variants

    def open(self):
        """Open HDF5 file and optional BigWig file for writing."""
        super().open()
        self.h5_writer.create_dataset(
            "variants",
            (len(self.variants),),
            dtype="S100",
            compression="gzip",
        )
        self.h5_writer.create_dataset(
            "rel_pos",
            (len(self.variants),),
            dtype="i4",
            compression="gzip",
        )
        self.h5_writer.create_dataset(
            "attribution_alt",
            (len(self.genes), 4, DECIMA_CONTEXT_SIZE),
            chunks=(1, 4, DECIMA_CONTEXT_SIZE),
            dtype="float32",
            compression="gzip",
        )
        self.h5_writer.create_dataset(
            "sequence_alt",
            (len(self.genes), DECIMA_CONTEXT_SIZE),
            chunks=(1, DECIMA_CONTEXT_SIZE),
            dtype="i1",
            compression="gzip",
        )
        self.h5_writer["variants"][:] = np.array(self.variants, dtype="S100")

    def add(
        self,
        variant: str,
        gene: str,
        rel_pos: int,
        seqs_ref: np.ndarray,
        attrs_ref: np.ndarray,
        seqs_alt: np.ndarray,
        attrs_alt: np.ndarray,
        gene_mask_start: int,
        gene_mask_end: int,
    ):
        """Add attribution data for a variant gene pair.

        Args:
            variant: Variant name from the variants list.
            gene: Gene name from the genes list.
            attrs: Attribution scores, shape (4, DECIMA_CONTEXT_SIZE).
            seqs: One-hot DNA sequence, shape (4, DECIMA_CONTEXT_SIZE).
        """
        super().add((variant, gene), seqs_ref, attrs_ref, gene_mask_start, gene_mask_end)
        idx = self.idx[(variant, gene)]
        self.h5_writer["variants"][idx] = np.array(variant, dtype="S100")
        self.h5_writer["rel_pos"][idx] = int(rel_pos)
        self.h5_writer["sequence_alt"][idx, :] = convert_input_type(
            torch.from_numpy(seqs_alt),  # convert_input_type only support Tensor
            "indices",
            input_type="one_hot",
        )[np.newaxis].astype("i1")
        self.h5_writer["attribution_alt"][idx, :, :] = attrs_alt.astype("float32")

    def close(self):
        """Close HDF5 file and optional BigWig file."""
        super().close()
        self.h5_writer.close()
