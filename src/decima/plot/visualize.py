import numpy as np
import pandas as pd
import plotnine as p9
import bioframe as bf

from ..tools.evaluate import match_criteria


def plot_gene_scatter(
    ad,
    gene,
    show_corr=True,
    show_abline=False,
    size=0.1,
    alpha=0.5,
    figure_size=(3, 2.5),
    corr_col="pearson",
    corrx=None,
    corry=None,
):
    # Extract data
    df = pd.DataFrame(
        {
            "true": ad[:, gene].X.squeeze(),
            "pred": ad[:, gene].layers["preds"].squeeze(),
        },
        index=ad.obs_names,
    )

    # Make plot
    g = (
        p9.ggplot(df, p9.aes(x="true", y="pred"))
        + p9.geom_pointdensity(size=size, alpha=alpha)
        + p9.theme_classic()
        + p9.theme(figure_size=figure_size)
        + p9.ggtitle(gene)
        + p9.xlab("True expression")
        + p9.ylab("Predicted expression")
        + p9.theme(plot_title=p9.element_text(face="italic"))
    )

    # Compute correlation
    if show_corr:
        corr = np.round(ad.var.loc[gene, corr_col], 2)
        corrx = corrx or 1
        corry = corry or df.pred.max() - 0.5
        g = g + p9.geom_text(x=corrx, y=corry, label=f"rho={corr}")

    if show_abline:
        # Add diagonal
        g = g + p9.geom_abline(intercept=0, slope=1)

    return g


def plot_track_scatter(
    ad,
    track,
    off_track=None,
    show_corr=True,
    show_abline=False,
    size=0.1,
    alpha=0.5,
    figure_size=(3, 2.5),
    corrx=None,
    corry=None,
):
    # Extract data
    df = pd.DataFrame(
        {
            "true": ad[track, :].X.squeeze(),
            "pred": ad[track, :].layers["preds"].squeeze(),
        },
        index=ad.var_names,
    )
    if off_track is not None:
        df["true"] = df["true"] - ad[off_track, :].X.squeeze()
        df["pred"] = df["pred"] - ad[off_track, :].layers["preds"].squeeze()

    # Make plot
    g = (
        p9.ggplot(df, p9.aes(x="true", y="pred"))
        + p9.geom_pointdensity(size=size, alpha=alpha)
        + p9.theme_classic()
        + p9.theme(figure_size=figure_size)
    )

    if off_track is None:
        g = g + p9.xlab("True expression") + p9.ylab("Predicted expression")
    else:
        g = g + p9.xlab("True log FC") + p9.ylab("Predicted log FC")

    # Compute correlation
    if show_corr:
        corr = np.round(df.corr().iloc[0, 1], 2)
        corrx = corrx or 1
        corry = corry or df.pred.max() - 0.5
        g = g + p9.geom_text(x=corrx, y=corry, label=f"rho={corr}")

    if show_abline:
        # Add diagonal
        g = g + p9.geom_abline(intercept=0, slope=1)

    return g


def plot_marker_box(
    gene,
    ad,
    marker_features,
    label_name="label",
    split_col=None,
    split_values=None,
    order=None,
    include_preds=True,
    fill=True,
):
    # Get criteria to filter
    if isinstance(marker_features, list):
        marker_features = ad.var.loc[gene, marker_features].to_dict()
    filter_df = pd.DataFrame(marker_features)

    # Collect observations for this gene
    to_plot = pd.DataFrame(
        {
            "True": ad[:, gene].X.squeeze(),
            "Predicted": ad[:, gene].layers["preds"].squeeze(),
            label_name: "Other",
        },
        index=ad.obs.index,
    )

    # Get matching observations
    labels = filter_df.apply(lambda row: "_".join(row.dropna()), axis=1).tolist()
    for i in range(len(filter_df)):
        match = match_criteria(df=ad.obs, filter_df=filter_df.iloc[[i]].copy())
        to_plot.loc[match, label_name] = labels[i]

    # Choose background organs
    if split_col is not None:
        to_plot[split_col] = ad.obs[split_col].tolist()
        split_values = split_values or to_plot[split_col].unique()
        for spl in split_values:
            to_plot.loc[
                (to_plot[split_col] == spl) & (to_plot[label_name] == "Other"),
                label_name,
            ] = f"Other {spl}"
        to_plot = to_plot.iloc[:, :3]

    if include_preds:
        # Reorder the factor levels based on the median value
        to_plot = to_plot.melt(id_vars=label_name)
        if order is None:
            order = to_plot.groupby(label_name)["value"].median().sort_values(ascending=False).index.tolist()
        to_plot[label_name] = pd.Categorical(to_plot[label_name], categories=order)

        # Plot
        to_plot.variable = pd.Categorical(to_plot.variable, categories=["True", "Predicted"])
        g = (
            p9.ggplot(to_plot, p9.aes(x="variable", y="value", fill=label_name))
            + p9.geom_boxplot(outlier_size=0.1)
            + p9.theme_classic()
            + p9.theme(figure_size=(3, 2.5))
            + p9.ggtitle(gene)
            + p9.ylab("Expression")
            + p9.theme(plot_title=p9.element_text(face="italic"))
            + p9.theme(axis_title_x=p9.element_blank())
        )

    else:
        # Reorder the factor levels based on the median value
        if order is None:
            order = to_plot.groupby(label_name)["True"].median().sort_values(ascending=False).index.tolist()
        to_plot[label_name] = pd.Categorical(to_plot[label_name], categories=order)

        # Plot
        if fill:
            g = p9.ggplot(to_plot, p9.aes(x=label_name, y="True", fill=label_name))
        else:
            g = p9.ggplot(to_plot, p9.aes(x=label_name, y="True"))
        g = (
            g
            + p9.geom_boxplot(outlier_size=0.1)
            + p9.theme_classic()
            + p9.theme(figure_size=(3, 2.5))
            + p9.ylab("Measured Expression")
            + p9.ggtitle(gene)
            + p9.theme(plot_title=p9.element_text(face="italic"))
            + p9.theme(axis_title_x=p9.element_blank())
            + p9.theme(axis_text_x=p9.element_text(rotation=60, hjust=1))
        )
    return g


def plot_peaks(attrs, tss_pos, df_peaks=None, overlapping_min_dist=1000, figsize=(10, 2)):
    """Plot peaks in attribution scores.

    Args:
        attr: Attribution scores array
        tss_pos: Position of TSS (relative to the gene)
        df_peaks: DataFrame containing peak information
        overlapping_min_dist: Minimum distance between peaks to consider them overlapping
        figsize: Figure size

    Returns:
        plotnine.ggplot: The plotted figure
    """
    to_plot = pd.DataFrame(
        {
            "distance from TSS": [x - tss_pos for x in range(attrs.shape[1])],
            "attribution": attrs.mean(0),
        }
    )
    g = (
        p9.ggplot(to_plot, p9.aes(x="distance from TSS", y="attribution"))
        + p9.geom_line()
        + p9.theme_classic()
        + p9.theme(figure_size=figsize)
    )

    if df_peaks is not None:
        df_peaks = df_peaks.copy()
        df_peaks["chrom"] = "_chr"
        for region in bf.merge(df_peaks, min_dist=overlapping_min_dist).itertuples():
            g += p9.annotate(
                xmin=region.start - tss_pos,
                xmax=region.end - tss_pos,
                ymin=to_plot["attribution"].min(),
                ymax=to_plot["attribution"].max(),
                geom="rect",
                color="#FF000040",
                fill="#FF000040",
            )

    return g
