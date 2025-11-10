"""Decima constants."""

import json
import os


DECIMA_CONTEXT_SIZE = 524_288
SUPPORTED_GENOMES = {"hg38"}
NUM_CELLS = 8856

DEFAULT_ENSEMBLE = "ensemble"
AVAILABLE_ENSEMBLES = [DEFAULT_ENSEMBLE]

ENSEMBLE_MODELS_NAMES = dict()

if "DECIMA_ENSEMBLE_MODELS_NAMES" in os.environ:
    ENSEMBLE_MODELS_NAMES = json.loads(os.environ["DECIMA_ENSEMBLE_MODELS_NAMES"])
else:
    ENSEMBLE_MODELS_NAMES["ensemble"] = ["v1_rep0", "v1_rep1", "v1_rep2", "v1_rep3"]

assert all(ensemble_name in AVAILABLE_ENSEMBLES for ensemble_name in ENSEMBLE_MODELS_NAMES.keys()), (
    f"Invalid ensemble names: {ENSEMBLE_MODELS_NAMES.keys()}. Available ensembles are: {AVAILABLE_ENSEMBLES}"
    "Check your `DECIMA_ENSEMBLE_MODELS_NAMES` environment variable if you are customizing the ensemble models."
)
