import logging
import genomepy
from decima.utils.resources import load_decima_model, load_decima_metadata


logger = logging.getLogger("decima")


def download_hg38():
    """Download hg38 genome from UCSC."""
    logger.info("Downloading hg38 genome...")
    genomepy.install_genome("hg38", provider="UCSC")


def download_decima_weights():
    """Download pre-trained Decima model weights from wandb."""
    logger.info("Downloading Decima model weights...")
    for rep in range(4):
        load_decima_model(rep)


def download_decima_metadata():
    """Download pre-trained Decima model data from wandb."""
    logger.info("Downloading Decima metadata...")
    load_decima_metadata()


def download_decima_data():
    """Download all required data for Decima."""
    download_hg38()
    download_decima_weights()
    download_decima_metadata()
