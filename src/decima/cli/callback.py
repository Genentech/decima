import click
from pathlib import Path
from decima.constants import MODEL_METADATA, ENSEMBLE_MODELS, DEFAULT_ENSEMBLE


def parse_model(ctx, param, value):
    if isinstance(value, int):
        value = str(value)

    if value is None:
        return None
    elif isinstance(value, str):
        if value in MODEL_METADATA:
            return value

        paths = value.split(",")
        for path in paths:
            if not Path(path).exists():
                raise click.ClickException(
                    f"Model path {path} does not exist. Check if the path is correct and the file exists."
                )
        return paths

    return value


def parse_genes(ctx, param, value):
    if value is None:
        return None
    elif isinstance(value, str):
        return value.split(",")
    raise ValueError(f"Invalid genes: {value}. Genes should be a comma-separated list of gene names or None.")


def validate_save_replicates(ctx, param, value):
    if value:
        if ctx.params["model"] in ENSEMBLE_MODELS:
            return value
        elif isinstance(ctx.params["model"], list) and (len(ctx.params["model"]) > 1):
            return value
        else:
            raise ValueError(
                "`--save-replicates` is only supported for ensemble models. Pass `ensemble` or list of models as the model argument."
            )
    return value


def parse_metadata(ctx, param, value):
    if value is None:
        if "model" in ctx.params:
            model = ctx.params["model"]
        else:
            model = DEFAULT_ENSEMBLE

        if isinstance(model, list):
            model = model[0]
        if Path(model).exists():
            raise click.ClickException(
                f"File path passed for model {model} but metadata filepath is not provided. Also, provide the metadata filepath."
            )
        else:
            return model
    elif isinstance(value, str):
        if value in MODEL_METADATA:
            return value
        elif Path(value).exists():
            return value
        else:
            raise click.ClickException(
                f"Invalid name for the metadata dataset: {value}. Check if the name is correct or the metadata file exists."
            )


def parse_attributions(ctx, param, value):
    value = value.split(",")
    for i in value:
        if not Path(i).exists():
            raise click.ClickException(
                f"Attribution path {i} does not exist. Check if the path is correct and the file exists."
            )
        elif not i.endswith(".h5"):
            raise click.ClickException(
                f"Attribution path {i} is not a h5 file. Check if the path is correct and the file is a h5 file."
            )
    return value
