from typing import Iterator
import h5py
from pathlib import Path
import numpy as np
import pyBigWig
import genomepy
import pandas as pd
from pyfaidx import Fasta
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
    df = list()
    with Fasta(fasta_file) as fasta:
        # fasta = Fasta(fasta_file)
        for record in fasta:
            name, start, end = record.name.split("|")
            start_label, start = start.split("=")
            end_label, end = end.split("=")
            assert (start_label == "gene_mask_start") and (end_label == "gene_mask_end"), (
                "First boundary label must be `gene_mask_start` and second boundary label must be `gene_mask_end` "
                "for example: `>seq|gene_mask_start=1|gene_mask_end=10`."
            )
            df.append({"gene": name, "seq": str(record[:]), "gene_mask_start": int(start), "gene_mask_end": int(end)})
        return pd.DataFrame(df).set_index("gene")


def read_vcf_chunks(vcf_file: str, chunksize: int) -> Iterator[pd.DataFrame]:
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
        self.sizes = genomepy.Genome(genome).sizes
        self.measures = dict()

    def open(self):
        """Open BigWig file for writing and add chromosome header."""
        self.bw = pyBigWig.open(self.path, "w")
        self.bw.addHeader([(chrom, size) for chrom, size in self.sizes.items()])

    def __enter__(self):
        """Context manager entry - opens BigWig file."""
        self.open()
        return self

    def add(self, chrom, start, end, values):
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
        for chrom, data in self.measures.items():
            values = data["values"] / data["count"]
            pos = np.where(np.abs(values) > self.threshold)[0]
            values = values[pos]
            if len(values) == 0:
                continue
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
        metadata_anndata: Gene metadata source. None uses default Decima data.
        genome: Reference genome version.
        bigwig: Create BigWig file for genome browser.

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

    def __init__(self, path, genes, model_name, metadata_anndata=None, genome: str = "hg38", bigwig: bool = True):
        self.path = path
        self.genes = genes
        self.genome = genome
        self.bigwig = bigwig
        self.model_name = model_name
        self.idx = {g: i for i, g in enumerate(self.genes)}
        self.result = DecimaResult.load(metadata_anndata)

    def open(self):
        """Open HDF5 file and optional BigWig file for writing."""
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
            (len(self.genes), 4, DECIMA_CONTEXT_SIZE),
            chunks=(1, 4, DECIMA_CONTEXT_SIZE),
            dtype="i1",
            compression="gzip",
        )
        if self.bigwig:
            self.bigwig_writer = BigWigWriter(Path(self.path).with_suffix(".bigwig").as_posix(), self.genome)
            self.bigwig_writer.open()

    def __enter__(self):
        """Context manager entry - opens files for writing."""
        self.open()
        return self

    def add(self, gene, seqs, attrs):
        """Add attribution data for a gene.

        Args:
            gene: Gene name from the genes list.
            attrs: Attribution scores, shape (4, DECIMA_CONTEXT_SIZE).
            seqs: One-hot DNA sequence, shape (4, DECIMA_CONTEXT_SIZE).
        """
        self.h5_writer["sequence"][self.idx[gene], :, :] = seqs.astype("i1")
        self.h5_writer["attribution"][self.idx[gene], :, :] = attrs.astype("float32")

        if self.bigwig:
            gene_meta = self.result.get_gene_metadata(gene)
            self.bigwig_writer.add(gene_meta.chrom, gene_meta.start, gene_meta.end, attrs.mean(axis=0))

    def close(self):
        """Close HDF5 file and optional BigWig file."""
        self.h5_writer.close()
        if self.bigwig:
            self.bigwig_writer.close()

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit - closes files."""
        self.close()
