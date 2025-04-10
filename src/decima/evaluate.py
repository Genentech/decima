import numpy as np
import pandas as pd
from scipy.stats import pearsonr, zscore
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm


def match_criteria(df, filter_df):
    for col in filter_df.columns:
        all_values = df[col].unique().tolist()
        isna = filter_df[col].isna()
        if isna.sum() > 0:
            filter_df.loc[isna, col] = [all_values] * isna.sum()
            filter_df = filter_df.explode(col)

    sel_idxs = df.reset_index().merge(filter_df, how="inner")["index"]
    return df.index.isin(sel_idxs)


def marker_zscores(ad, key="cell_type", layer=None):
    E = ad.X if layer is None else ad.layers[layer]
    z = zscore(E, axis=0)

    dfs = []
    for group in tqdm(ad.obs[key].unique()):
        scores = z[ad.obs[key] == group, :].mean(0).squeeze()
        df = pd.DataFrame({"gene": ad.var_names, "score": scores, key: group})
        dfs.append(df)

    return pd.concat(dfs, axis=0).reset_index(drop=True)


def compare_marker_zscores(ad, key="cell_type"):
    marker_df_obs = marker_zscores(ad, key)
    marker_df_pred = marker_zscores(ad, key, layer="preds")
    marker_df = marker_df_pred.merge(marker_df_obs, on=["gene", key], suffixes=("_pred", "_obs"))
    return marker_df


def compute_marker_metrics(marker_df, key="cell_type", tp_cutoff=1):
    df_list = []
    for k in set(marker_df[key]):
        # get celltype data
        curr_marker_df = marker_df[marker_df[key] == k].copy()

        # compute corrs
        corr = pearsonr(curr_marker_df["score_obs"], curr_marker_df["score_pred"])[0]

        # compute binary labels
        labels = curr_marker_df["score_obs"] > tp_cutoff
        n_positive = np.sum(labels)

        if n_positive == len(labels) or n_positive == 0:
            auprc, auroc = np.nan
        else:
            auprc = average_precision_score(labels, curr_marker_df["score_pred"])
            auroc = roc_auc_score(labels, curr_marker_df["score_pred"])

        # append df
        curr_df = pd.DataFrame(
            {
                key: [k],
                "n_positive": [n_positive],
                "pearson": [corr],
                "auprc": [auprc],
                "auroc": [auroc],
            }
        )
        df_list.append(curr_df)

    return pd.concat(df_list)
