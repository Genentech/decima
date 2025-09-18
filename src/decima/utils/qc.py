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
        """Log a message

        Args:
            message (str): Message to log
            level (str, optional): Level of the message. Defaults to "info".
        """
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

    def log_gene(self, gene: str, threshold: float = 0.5):
        """Log the correlation of a gene with the model

        Args:
            gene (str): Gene to log
            threshold (float, optional): Threshold for logging. Defaults to 0.5.
        """
        gene_metadata = self.result.get_gene_metadata(gene)
        if gene_metadata.pearson < threshold:
            self.log(
                f"Gene {gene} has low correlation with the model. Pearson: {gene_metadata.pearson}. "
                "Be careful with the predictions of the model for this gene. "
                "Check `DecimaResult.load().gene_metadata['pearson']` to see the correlation of the gene with the model.",
                level="warning",
            )

    def log_correlation(self, tasks, off_tasks=None, plot=True):
        """Log the correlation between tasks and off_tasks

        Args:
            tasks (str): Tasks to use for correlation
            off_tasks (str): Off tasks to use for correlation
            plot (bool, optional): Whether to plot the correlation. Defaults to True.
        """
        pearsonr = self.result.correlation(tasks, off_tasks)
        self.log(
            f"Correlation between tasks and off_tasks: {pearsonr[0]:.2f} (P={pearsonr[1]:.2e})",
        )
        if plot:
            g = self.result.plot_correlation(tasks, off_tasks)
            g.save(self.log_file.replace(".warnings.qc.log", "_correlation.qc.png"))

    def open(self):
        self._log_file = open(self.log_file, "w")

    def close(self):
        self._log_file.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
