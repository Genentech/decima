import json

import anndata
import bioframe as bf
import numpy as np
import pandas as pd
from grelu.io.genome import read_sizes
from grelu.sequence.format import intervals_to_strings
from grelu.sequence.utils import get_unique_length
from tqdm import tqdm
import mygene


from decima.constants import DECIMA_CONTEXT_SIZE


def merge_transcripts(gtf):
    # Get gene-level columns
    genes = gtf[["chrom", "start", "end", "strand", "gene_id", "gene_type", "gene_name"]].copy()

    # Aggregate all features from the same gene
    genes = genes.groupby("gene_name").agg(lambda x: list(set(x)))

    # Get minimum start and maximum end
    genes["start"] = genes["start"].apply(min)
    genes["end"] = genes["end"].apply(max)

    # Merge all other columns
    for col in ["chrom", "strand", "gene_id", "gene_type"]:
        genes[col] = genes[col].apply(lambda x: x[0])

    return genes


def var_to_intervals(ad, chr_end_pad=10000, genome="hg38", seq_len=DECIMA_CONTEXT_SIZE, crop_coords=163840):
    sizes = read_sizes(genome)

    # Calculate interval size
    print(
        f"The interval size is {seq_len} bases. Of these, {crop_coords} will be upstream of the gene start and {seq_len - crop_coords} will be downstream of the gene start."
    )

    # Create intervals around + strand genes
    ad.var.loc[ad.var.strand == "+", "start"] = ad.var.loc[ad.var.strand == "+", "gene_start"] - crop_coords
    ad.var.loc[ad.var.strand == "+", "end"] = ad.var.loc[ad.var.strand == "+", "start"] + seq_len

    # Create interval around - strand genes
    ad.var.loc[ad.var.strand == "-", "end"] = ad.var.loc[ad.var.strand == "-", "gene_end"] + crop_coords
    ad.var.loc[ad.var.strand == "-", "start"] = ad.var.loc[ad.var.strand == "-", "end"] - seq_len

    # shift sequences with start < 0
    crossing_start = ad.var.start < chr_end_pad
    ad.var.loc[crossing_start, "start"] = chr_end_pad
    ad.var.loc[crossing_start, "end"] = ad.var.loc[crossing_start, "start"] + seq_len
    print(f"{np.sum(crossing_start)} intervals extended beyond the chromosome start and have been shifted")

    # shift sequences with end > chromosome size
    crossing_end = 0
    for chrom, size in sizes.values:
        max_end = size - chr_end_pad
        drop = (ad.var.chrom == chrom) & (ad.var.end > max_end)
        crossing_end += drop.sum()
        ad.var.loc[drop, "end"] = max_end
        ad.var.loc[drop, "start"] = ad.var.loc[drop, "end"] - seq_len
    print(f"{crossing_end} intervals extended beyond the chromosome end and have been shifted")

    # gene start position on output sequence
    ad.var.loc[ad.var.strand == "+", "gene_mask_start"] = (
        ad.var.loc[ad.var.strand == "+", "gene_start"] - ad.var.loc[ad.var.strand == "+", "start"]
    )
    ad.var.loc[ad.var.strand == "-", "gene_mask_start"] = (
        ad.var.loc[ad.var.strand == "-", "end"] - ad.var.loc[ad.var.strand == "-", "gene_end"]
    )
    ad.var.gene_mask_start = ad.var.gene_mask_start.astype(int)
    ad.var.gene_length = ad.var.gene_length.astype(int)

    # Get gene end position on sequence
    ad.var["gene_mask_end"] = (ad.var.gene_mask_start + ad.var.gene_length).apply(lambda x: min(seq_len, x))
    ad.var.gene_mask_end = ad.var.gene_mask_end.astype(int)

    # Drop intervals with less than crop_coords upstream bases
    drop = ad.var.gene_mask_start < crop_coords
    ad = ad[:, ~drop]
    print(f"{np.sum(drop)} intervals did not extend far enough upstream of the TSS and have been dropped")

    # Check length
    assert get_unique_length(ad.var) == seq_len
    return ad


def assign_borzoi_folds(ad, splits):
    # Extract gene intervals
    genes = ad.var.reset_index().rename(columns={"index": "gene_name"})

    # Overlap with Borzoi splits
    overlaps = bf.overlap(genes, splits, how="left")
    overlaps = overlaps[["gene_id", "fold_"]].drop_duplicates()
    overlaps.columns = ["gene_id", "fold"]

    # List all overlapping folds for each gene
    overlaps = (
        overlaps.groupby("gene_id")
        .fold.apply(list)
        .apply(lambda x: x if x[0] is None else ",".join([f[-1] for f in x]))
    )
    overlaps = overlaps.reset_index()
    overlaps.loc[overlaps.fold.apply(lambda x: x[0] is None), "fold"] = "none"

    # Add back to AnnData
    ind = ad.var.index
    ad.var = ad.var.merge(overlaps, on="gene_id", how="left")
    ad.var.index = ind
    return ad


def aggregate_anndata(
    ad,
    by_cols=[
        "cell_type",
        "tissue",
        "organ",
        "disease",
        "study",
        "dataset",
        "region",
        "subregion",
        "celltype_coarse",
    ],
    sum_cols=["n_cells"],
):
    # Get column names
    obs_cols = by_cols + sum_cols
    gene_names = ad.var_names.tolist()

    # Format obs
    print("Creating new obs matrix")
    obs = ad.obs[obs_cols].copy()
    for col in by_cols:
        obs[col] = obs[col].astype(str)
    for col in sum_cols:
        obs[col] = obs[col].astype(int)

    # Create X
    X = pd.DataFrame(ad.X, index=obs.index.tolist(), columns=gene_names)
    X = pd.concat(
        [
            obs,
            X,
        ],
        axis=1,
    )

    # Aggregate X
    print("Aggregating")
    X = X.groupby(by_cols).sum().reset_index()

    # Split off the obs again
    obs = X[obs_cols]
    obs.index = [f"agg_{i}" for i in range(len(obs))]
    X = X[gene_names]

    # Create new anndata
    print("Creating new anndata")
    new_ad = anndata.AnnData(X=np.array(X), obs=obs, var=ad.var.copy())
    return new_ad


def change_values(df, col, value_dict):
    df[col] = df[col].astype(str)
    for k, v in value_dict.items():
        df.loc[df[col] == k, col] = v
    return df


def get_frac_N(interval, genome="hg38"):
    seq = intervals_to_strings(interval, genome=genome)
    return seq.count("N") / len(seq)


def match_cellranger_2024(ad, genes24):
    matched = 0
    unmatched_genes = ad.var.index[ad.var.gene_id.isna()].tolist()
    print(f"{len(unmatched_genes)} genes unmatched.")
    for gene in tqdm(unmatched_genes):
        if gene in genes24.index.tolist():
            gene_id = genes24.gene_id[genes24.index == gene].values[0]
            if gene_id not in ad.var.gene_id.tolist():
                for col in ad.var.columns:
                    ad.var.loc[gene, col] = genes24.loc[genes24.index == gene, col].values[0]
                matched += 1

    print(f"{matched} genes matched.")


def match_ref_ad(ad, ref_ad):
    matched = 0
    unmatched_genes = ad.var.index[ad.var.gene_id.isna()].tolist()
    print(f"{len(unmatched_genes)} genes unmatched.")

    for gene in tqdm(unmatched_genes):
        if gene in ref_ad.var.index.tolist():
            gene_id = ref_ad.var.gene_id[ref_ad.var.index == gene].values[0]
            if gene_id not in ad.var.gene_id.tolist():
                for col in ad.var.columns:
                    ad.var.loc[gene, col] = ref_ad.var.loc[ref_ad.var.index == gene, col].values[0]
                matched += 1

    print(f"{matched} genes matched.")


def load_ncbi_string(string, allow_dups=False, verbose=False):
    out = []
    reports = json.loads(string[0])

    # Check the total count
    if reports == {"total_count": 0}:
        pass
    else:
        for i, r in enumerate(reports["reports"]):
            try:
                curr_dict = {}

                # Get gene metadata
                curr_dict["gene_name"] = r["query"][0] if "query" in r else r["gene"]["symbol"]
                r = r["gene"]
                curr_dict["symbol"] = r["symbol"]
                curr_dict["chrom"] = "chr" + r["chromosomes"][0]
                curr_dict["gene_type"] = r["type"]

                # Get ensembl ID
                if "ensembl_gene_ids" in r:
                    eids = r["ensembl_gene_ids"]
                    assert len(eids) == 1
                    curr_dict["gene_id"] = eids[0]
                else:
                    curr_dict["gene_id"] = r["symbol"]

                # Get genomic position
                annot = [x for x in r["annotations"] if x["assembly_name"] == "GRCh38.p14"][0]
                loc = annot["genomic_locations"]
                assert len(loc) == 1
                loc = loc[0]["genomic_range"]

                curr_dict["start"] = loc["begin"]
                curr_dict["end"] = loc["end"]
                curr_dict["strand"] = "-" if loc["orientation"] == "minus" else "+"

                out.append(curr_dict)

            except Exception as e:
                if verbose:
                    print(i, str(e))
                else:
                    pass

        out = pd.DataFrame(out)
        if not allow_dups:
            out = out[out.gene_name.isin(out.gene_name.value_counts()[out.gene_name.value_counts() == 1].index)]
        out["source"] = "ncbi"
        return out


def match_ncbi(ad, ncbi):
    matched = 0
    unmatched_genes = ad.var.index[ad.var.gene_id.isna()].tolist()
    print(f"{len(unmatched_genes)} genes unmatched.")
    for gene in tqdm(unmatched_genes):
        if gene in ncbi.gene_name.tolist():
            for col in ad.var.columns:
                ad.var.loc[gene, col] = ncbi.loc[ncbi.gene_name == gene, col].values[0]
            matched += 1

    print(f"{matched} genes matched.")


def make_inputs(gene, ad):
    raise DeprecationWarning(
        "make_inputs() is deprecated and will be removed in a future version. "
        "Please use the DecimaResult.prepare_one_hot() directly instead."
    )


def return_ensembl(queries, on="gene_id"):
    mg = mygene.MyGeneInfo()
    df = []
    fields = [
        "ensembl.gene",
        "symbol",
        "type_of_gene",
        "genomic_pos.chr",
        "genomic_pos.start",
        "genomic_pos.end",
        "genomic_pos.strand",
    ]
    for query in tqdm(queries):
        # Make the query
        if on == "gene_id":
            hits = mg.query(f"ensembl.gene:{query}", fields=fields)["hits"]
            hits = [hit for hit in hits if ("symbol" in hit) and ("ensembl" in hit)]

        elif on == "gene_name":
            hits = mg.query(query, fields=fields)["hits"]
            hits = [hit for hit in hits if ("symbol" in hit) and ("ensembl" in hit)]
            hits = [hit for hit in hits if len(hit["ensembl"]) == 1]
            hits = [hit for hit in hits if hit["ensembl"]["gene"].startswith("ENSG")]
            if len(hits) > 1:
                hits = [hit for hit in hits if hit["symbol"] == query]

        # Format the query
        if len(hits) == 1:
            hit = hits[0]
            if type(hit["genomic_pos"]) is dict:
                hit["chrom"] = hit["genomic_pos"]["chr"]
                for col in ["start", "end", "strand"]:
                    hit[col] = hit["genomic_pos"][col]
                del hit["genomic_pos"]

                if on == "gene_id":
                    hit["gene_id"] = query
                elif on == "gene_name":
                    hit["gene_id"] = hit["ensembl"]["gene"]
                    hit["gene_name"] = query

                del hit["ensembl"]
                df.append(pd.DataFrame.from_dict(hit, orient="index").T)

    df = pd.concat(df).reset_index(drop=True)
    df.strand = df.strand.map({1: "+", -1: "-"})
    df["source"] = "mygene"
    return df
