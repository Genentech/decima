import logging
from decima.core.result import DecimaResult


class QCLogger:
    """
    Logger for QC

    Args:
        log_file (str): Path to the log file
        metadata_anndata (str): Path to the metadata anndata file
    """

    def __init__(self, log_file: str, metadata_anndata: str = None):
        self.log_file = log_file
        self.result = DecimaResult.load(metadata_anndata)

    def log(self, message: str, level: str = "info"):
        logger = logging.getLogger("decima")
        if level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        else:
            raise ValueError(f"Invalid level: {level}")

        self._log_file.write(f"{level}: {message}\n")

    def open(self):
        self._log_file = open(self.log_file, "w")

    def close(self):
        self._log_file.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def log_gene(self, gene: str, threshold: float = 0.7):
        gene_metadata = self.result.get_gene_metadata(gene)
        if gene_metadata.pearson < threshold:
            self.log(
                f"Gene {gene} has low correlation with the model. Pearson: {gene_metadata.pearson}. "
                "Be careful with the predictions of the model for this gene. "
                "Check `DecimaResult.load().gene_metadata['pearson']` to see the correlation of the gene with the model.",
                level="warning",
            )

    def log_correlation(self, tasks, off_tasks, layer="preds"):
        # TODO: compare the correlation between tasks and off_tasks and log the result if the correlation is low.
        raise NotImplementedError("Not implemented yet")

    def plot_correlation(self, tasks, off_tasks, layer="preds"):
        # TODO: plot the correlation between tasks and off_tasks and save the plot
        raise NotImplementedError("Not implemented yet")
