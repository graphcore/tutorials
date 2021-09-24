from click.testing import Result


def print_exception(result: Result):
    if result.exception or result.exit_code != 0:
        print(result.exception)
        print(result.stdout)
        print(result.stderr)
