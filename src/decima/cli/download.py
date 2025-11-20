"""
Download CLI.

This module contains the CLI for downloading the required data and model weights.

`decima download` is the main command for downloading the required data and model weights.

It includes subcommands for:
- Caching the required data and model weights. `cache`
"""

import click
from decima.constants import DEFAULT_ENSEMBLE
from decima.cli.callback import parse_model
from decima.hub.download import (
    cache_decima_data,
    download_decima_weights,
    download_decima_metadata,
    download_decima,
)


@click.command()
def cli_cache():
    """Cache all required data and model weights."""
    cache_decima_data()


@click.command()
@click.option(
    "--model",
    type=str,
    default=DEFAULT_ENSEMBLE,
    help=f"Model to download. Default: {DEFAULT_ENSEMBLE}.",
    callback=parse_model,
)
@click.option(
    "--download-dir",
    type=click.Path(),
    default=".",
    help="Directory to download the model weights. Default: current directory.",
)
def cli_download_weights(model, download_dir):
    """Download pre-trained Decima model weights."""
    download_decima_weights(model, str(download_dir))


@click.command()
@click.option(
    "--metadata",
    default=DEFAULT_ENSEMBLE,
    help=f"Model to download metadata for using wandb. Default: {DEFAULT_ENSEMBLE}.",
    callback=parse_model,
)
@click.option(
    "--download-dir",
    type=click.Path(),
    default=".",
    help="Directory to download the metadata. Default: current directory.",
)
def cli_download_metadata(metadata=DEFAULT_ENSEMBLE, download_dir=None):
    """Download pre-trained Decima metadata for a given model."""
    download_decima_metadata(metadata, str(download_dir))


@click.command()
@click.option(
    "--model-name",
    type=str,
    default=DEFAULT_ENSEMBLE,
    help=f"Model to download. Default: {DEFAULT_ENSEMBLE}.",
    callback=parse_model,
)
@click.option(
    "--download-dir", type=click.Path(), default=".", help="Directory to download the data. Default: current directory."
)
def cli_download(model_name=DEFAULT_ENSEMBLE, download_dir="."):
    """Download model weights and metadata for Decima."""
    download_decima(model_name, str(download_dir))
