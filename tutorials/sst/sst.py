import click


@click.group()
def cli():
    pass


@cli.command()
@click.option('--filename', '-f', required=True, type=click.STRING,
              help='Absolute or relative path to python file to be converted')
def py2ipynb(filename: click.STRING) -> None:
    """
    Description
    """
    print(filename)


if __name__ == '__main__':
    cli()
