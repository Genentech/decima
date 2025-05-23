import pandas as pd
from pyfaidx import Fasta


def read_fasta(fasta_file: str) -> pd.DataFrame:
    df = list()
    with Fasta(fasta_file) as fasta:
        for name, seq in fasta.items():
            name = name.split(" ")
            genes = name[1:]
            genes = [g.split("=")[1] for g in genes]

            df.append({"seq": seq, "gene_mask_start": 0, "gene_mask_end": len(seq)})
    return pd.DataFrame(df)
