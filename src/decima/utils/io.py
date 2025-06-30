from typing import Iterator
import pandas as pd
from pyfaidx import Fasta


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
