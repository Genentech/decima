import os


DECIMA_CONTEXT_SIZE = 524288
SUPPORTED_GENOMES = {"hg38"}
NUM_CELLS = 8856

if "DECIMA_ENSEMBLE_MODELS_NAMES" in os.environ:
    ENSEMBLE_MODELS_NAMES = os.environ["DECIMA_ENSEMBLE_MODELS_NAMES"].split(",")
else:
    ENSEMBLE_MODELS_NAMES = ["v1_rep0", "v1_rep1", "v1_rep2", "v1_rep3"]
