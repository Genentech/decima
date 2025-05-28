import pandas as pd
from grelu.sequence.format import one_hot_to_strings
from decima.core import DecimaResult


result = DecimaResult.load()

df = list()

# for i in ['CD68', 'SPI1', 'CD14']:
for i in ['CD68', 'SPI1']:
    seq, _ = result.prepare_one_hot(i)
    seq = one_hot_to_strings(seq)
    gene = result.gene_metadata.loc[i]

    df.append({
        'name': gene.name,
        'seq': seq,
        'gene_mask_start': gene.gene_mask_start,
        'gene_mask_end': gene.gene_mask_end
    })

df = pd.DataFrame(df).set_index('name')
df.to_csv("tests/data/seqs.csv")

with open("tests/data/seqs.fasta", "w") as f:
    for row in df.itertuples():
        f.write(f">{row.Index}|gene_mask_start={row.gene_mask_start}|gene_mask_end={row.gene_mask_end}\n{row.seq}\n")
