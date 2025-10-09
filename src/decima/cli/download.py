"""
Download CLI.

This module contains the CLI for downloading the required data and model weights.

`decima download` is the main command for downloading the required data and model weights.

It includes subcommands for:
- Downloading the required data and model weights. `download`
"""

import click
from decima.hub.download import download_decima_data


@click.command()
def cli_download():
    """Download all required data and model weights."""
    download_decima_data()
