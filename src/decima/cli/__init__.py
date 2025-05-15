import click
from decima.cli.finetune import finetune
from decima.cli.predict_genes import predict_genes
from decima.cli.download import download
from decima.cli.attributions import attributions



@click.group()
def main():
    """Decima CLI interface."""
    pass

main.add_command(finetune)
main.add_command(predict_genes)
# TODO: discuss if we want to add download command
# main.add_command(download)
main.add_command(attributions)

if __name__ == "__main__":
    main()