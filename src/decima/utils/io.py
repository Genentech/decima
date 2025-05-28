import pandas as pd
from pyfaidx import Fasta


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
