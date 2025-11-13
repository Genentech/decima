"""Decima constants."""

import json
import os


# constants for all models
DECIMA_CONTEXT_SIZE = 524_288
SUPPORTED_GENOMES = {"hg38"}

# EDIT: following metadata to add new models;
# metadata of models models
# models has dict as values
# ensemble models have list of model names as values and fetched metadata from the models
# following fields are required in the metadata:
# - name of the models on wandb
# - number of cells of the model
# - metadata name in the wandb
# - model_path [optional] to the local model path
# - metadata_path [optional] to the local metadata path
MODEL_METADATA = {
    "v1_rep0": {"name": "rep0", "num_tasks": 8856, "metadata": "metadata"},
    "v1_rep1": {"name": "rep1", "num_tasks": 8856, "metadata": "metadata"},
    "v1_rep2": {"name": "rep2", "num_tasks": 8856, "metadata": "metadata"},
    "v1_rep3": {"name": "rep3", "num_tasks": 8856, "metadata": "metadata"},
    "ensemble": ["v1_rep0", "v1_rep1", "v1_rep2", "v1_rep3"],
}
MODEL_METADATA["rep0"] = MODEL_METADATA["v1_rep0"]
MODEL_METADATA["rep1"] = MODEL_METADATA["v1_rep1"]
MODEL_METADATA["rep2"] = MODEL_METADATA["v1_rep2"]
MODEL_METADATA["rep3"] = MODEL_METADATA["v1_rep3"]
MODEL_METADATA[0] = MODEL_METADATA["v1_rep0"]
MODEL_METADATA[1] = MODEL_METADATA["v1_rep1"]
MODEL_METADATA[2] = MODEL_METADATA["v1_rep2"]
MODEL_METADATA[3] = MODEL_METADATA["v1_rep3"]
MODEL_METADATA["0"] = MODEL_METADATA["v1_rep0"]
MODEL_METADATA["1"] = MODEL_METADATA["v1_rep1"]
MODEL_METADATA["2"] = MODEL_METADATA["v1_rep2"]
MODEL_METADATA["3"] = MODEL_METADATA["v1_rep3"]

# default version
DEFAULT_ENSEMBLE = "ensemble"

# overwrite model metadata from environment variables
if "MODEL_METADATA" in os.environ:
    MODEL_METADATA = json.loads(os.environ["MODEL_METADATA"])

if "DEFAULT_ENSEMBLE" in os.environ:
    DEFAULT_ENSEMBLE = os.environ["DEFAULT_ENSEMBLE"]

ENSEMBLE_MODELS = [k for k, v in MODEL_METADATA.items() if isinstance(v, list)]
