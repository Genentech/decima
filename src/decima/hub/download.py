import os
from pathlib import Path
from typing import Union
import logging
import genomepy
from grelu.resources import get_artifact, DEFAULT_WANDB_HOST
from decima.constants import DEFAULT_ENSEMBLE, ENSEMBLE_MODELS, MODEL_METADATA
from decima.hub import login_wandb, load_decima_model, load_decima_metadata


logger = logging.getLogger("decima")


def cache_hg38():
    """Download hg38 genome from UCSC."""
    logger.info("Downloading hg38 genome...")
    genomepy.install_genome(provider="url", name="http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz")


def cache_decima_weights():
    """Download pre-trained Decima model weights from wandb."""
    logger.info("Downloading Decima model weights...")
    for rep in MODEL_METADATA[DEFAULT_ENSEMBLE]:
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


def download_decima_weights(model: Union[str, int] = DEFAULT_ENSEMBLE, download_dir: str = "."):
    """Download pre-trained Decima model weights from wandb.

    Args:
        model_name: Model name or replicate number.
        download_dir: Directory to download the model weights.

    Returns:
        Path to the downloaded model weights.
    """
    if model in ENSEMBLE_MODELS:
        return [download_decima_weights(model, download_dir) for model in MODEL_METADATA[model]]

    model_name = MODEL_METADATA[model]["name"]
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading Decima model weights for {model} to {download_dir / f'{model_name}.safetensors'}")

    art = get_artifact(model_name, project="decima", host=os.environ.get("WANDB_HOST", DEFAULT_WANDB_HOST))
    art.download(str(download_dir))
    return download_dir / f"{model_name}.safetensors"


def download_decima_metadata(metadata: str = DEFAULT_ENSEMBLE, download_dir: str = "."):
    """Download pre-trained Decima model data from wandb.

    Args:
        download_dir: Directory to download the metadata.
        metadata: Name of the model to download metadata for using wandb.

    Returns:
        Path to the downloaded metadata.
    """
    metadata = metadata or DEFAULT_ENSEMBLE
    if metadata in ENSEMBLE_MODELS:
        metadata = MODEL_METADATA[metadata][0]

    metadata_name = MODEL_METADATA[metadata]["metadata"]
    art = get_artifact(metadata_name, project="decima", host=os.environ.get("WANDB_HOST", DEFAULT_WANDB_HOST))
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading Decima metadata to {download_dir / f'{metadata_name}.h5ad'}.")

    art.download(str(download_dir))
    return download_dir / f"{metadata_name}.h5ad"


def download_decima(model: str = DEFAULT_ENSEMBLE, download_dir: str = "."):
    """Download all required data for Decima.

    Args:
        download_dir: Directory to download the model weights and metadata.

    Returns:
        Path to the downloaded directory containing the model weights and metadata.
    """
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading Decima model weights and metadata to {download_dir}:")

    download_decima_weights(model, download_dir)
    download_decima_metadata(model, download_dir)
    return download_dir
