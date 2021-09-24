from click.testing import Result


def print_exception(result: Result):
    if result.exception or result.exit_code != 0:
        if result.exception:
            print(result.exception)
        if result.stdout:
            print(result.stdout)
