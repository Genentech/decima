import logging
import click

# from decima.cli.finetune import finetune
from decima.cli.predict_genes import predict_genes
from decima.cli.download import download
from decima.cli.attributions import attributions
from decima.cli.query_cell import query_cell


logger = logging.getLogger("decima")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


@click.group()
def main():
    """Decima CLI interface."""
    pass


# main.add_command(finetune)
main.add_command(predict_genes)
main.add_command(download)
main.add_command(attributions)
main.add_command(query_cell)

if __name__ == "__main__":
    main()
