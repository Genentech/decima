import genomepy


def download_hg38():
    """Download hg38 genome from UCSC."""
    genomepy.install_genome("hg38", provider="UCSC")


def download_borzoi_weights():
    """Download pre-trained Borzoi model weights from wandb."""
    raise NotImplementedError("Downloading Borzoi weights from wandb is not yet implemented")


def download_decima_weights():
    """Download pre-trained Decima model weights from wandb."""
    raise NotImplementedError("Downloading Decima weights from wandb is not yet implemented")


def download_decima_metadata():
    """Download pre-trained Decima model data from wandb."""
    raise NotImplementedError("Downloading Decima data from wandb is not yet implemented")


def download_decima_data():
    """Download all required data for Decima."""
    download_hg38()
    download_borzoi_weights() 
    download_decima_weights()
    download_decima_metadata()