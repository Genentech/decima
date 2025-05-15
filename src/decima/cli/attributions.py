import click
from decima.interpret import predict_save_attributions



@click.command()
@click.option("-o", "--output_dir", type=str, required=True)
@click.option("--gene", type=str, required=True)
@click.option("--cells", type=str, required=True)
@click.option("--constract_cells", type=str, required=False)
def attributions(output_dir, gene, cells, constract_cells):
    predict_save_attributions(output_dir, gene, cells, constract_cells)