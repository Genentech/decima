import click
from decima.hub.download import download_decima_data


@click.command()
def cli_download():
    """Download all required data and model weights."""
    download_decima_data()
