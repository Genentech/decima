import click
from pathlib import Path


def parse_model(ctx, param, value):
    if value is None:
        return None
    elif isinstance(value, str):
        if value == "ensemble":
            return "ensemble"
        elif value in ["0", "1", "2", "3"]:
            return int(value)

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
        if ctx.params["model"] == "ensemble":
            return value
        elif isinstance(ctx.params["model"], str) and (len(ctx.params["model"].split(",")) > 1):
            return value
        else:
            raise ValueError(
                "`--save-replicates` is only supported for ensemble models. Pass `ensemble` or list of models as the model argument."
            )
    return value


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
