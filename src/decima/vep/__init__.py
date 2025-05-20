import pandas as pd
from decima.data.dataset import VariantDataset
from decima.model import load_model

from grelu.transforms.prediction_transforms import Aggregate


def predict_variant_effect(
    df_variant,
    tasks=None,
    model=0,
    batch_size=8,
    num_workers=16,
    device="cpu",
    include_cols=None,
    min_from_end=0,
    max_dist_tss=float("inf"),
):
    """Predict variant effect on a set of variants

    Args:
        df_variant (pd.DataFrame): DataFrame with variant information
        tasks (str, optional): Tasks to predict. Defaults to None.
        model (int, optional): Model to use. Defaults to 0.
        batch_size (int, optional): Batch size. Defaults to 8.
        num_workers (int, optional): Number of workers. Defaults to 16.
        device (str, optional): Device to use. Defaults to "cpu".
        include_cols (list, optional): Columns to include in the output. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with variant effect predictions
    """

    dataset = VariantDataset(
        df_variant, include_cols=include_cols, min_from_end=min_from_end, max_dist_tss=max_dist_tss
    )
    model = load_model(model=model, device=device)

    if tasks is not None:
        tasks = dataset.result.query_cells(tasks)
        agg_transform = Aggregate(tasks=tasks, model=model)
        model.add_transform(agg_transform)
    else:
        tasks = dataset.result.cells

    preds = model.predict_on_dataset(dataset, devices=device, batch_size=batch_size, num_workers=num_workers)

    df = dataset.variants
    df_pred = pd.DataFrame(preds, index=df.index, columns=tasks)

    return pd.concat([df, df_pred], axis=1)


# def predict_variant_effect_save(
#     df_variant,
#     tasks=None,
#     model=0,
#     batch_size=8,
#     num_workers=16,
#     device="cpu",
#     include_cols=None,
# ):


#     # df_pred = predict_variant_effect(df_variant, tasks, model, batch_size, num_workers, device, include_cols)
#     # df_pred.to_parquet(output_pq)
