from pathlib import Path
from typing import Union
import logging
import genomepy
from grelu.resources import get_artifact
from decima.constants import DEFAULT_ENSEMBLE, AVAILABLE_ENSEMBLES, ENSEMBLE_MODELS_NAMES
from decima.hub import login_wandb, load_decima_model, load_decima_metadata


logger = logging.getLogger("decima")


def cache_hg38():
    """Download hg38 genome from UCSC."""
    logger.info("Downloading hg38 genome...")
    genomepy.install_genome(provider="url", name="http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz")


def cache_decima_weights():
    """Download pre-trained Decima model weights from wandb."""
    logger.info("Downloading Decima model weights...")
    for rep in range(4):
        load_decima_model(rep)


def cache_decima_metadata():
    """Download pre-trained Decima model data from wandb."""
    logger.info("Downloading Decima metadata...")
    load_decima_metadata()


def cache_decima_data():
    """Download all required data for Decima."""
    login_wandb()
    cache_hg38()
    cache_decima_weights()
    cache_decima_metadata()


def download_decima_weights(model_name: Union[str, int], download_dir: str, ensemble: str = DEFAULT_ENSEMBLE):
    """Download pre-trained Decima model weights from wandb.

    Args:
        model_name: Model name or replicate number.
        download_dir: Directory to download the model weights.

    Returns:
        Path to the downloaded model weights.
    """
    if DEFAULT_ENSEMBLE in AVAILABLE_ENSEMBLES:
        return [download_decima_weights(model, download_dir) for model in range(4)]

    if model_name in {0, 1, 2, 3}:
        model_name = ENSEMBLE_MODELS_NAMES[ensemble][model_name]

    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading Decima model weights for {model_name} to {download_dir / f'{model_name}.safetensors'}")

    art = get_artifact(model_name, project="decima")
    art.download(str(download_dir))
    return download_dir / f"{model_name}.safetensors"


def download_decima_metadata(download_dir: str):
    """Download pre-trained Decima model data from wandb.

    Args:
        download_dir: Directory to download the metadata.

    Returns:
        Path to the downloaded metadata.
    """
    art = get_artifact("metadata", project="decima")
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading Decima metadata to {download_dir / 'metadata.h5ad'}.")

    art.download(str(download_dir))
    return download_dir / "metadata.h5ad"


def download_decima(download_dir: str):
    """Download all required data for Decima.

    Args:
        download_dir: Directory to download the model weights and metadata.

    Returns:
        Path to the downloaded directory containing the model weights and metadata.
    """
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading Decima model weights and metadata to {download_dir}:")

    download_decima_weights(DEFAULT_ENSEMBLE, download_dir)
    download_decima_metadata(download_dir)
    return download_dir
