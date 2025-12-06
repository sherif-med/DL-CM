import click

from . import (
    chain_decorators,
    checkpoint_path_option,
    config_file_option,
)

BasePredictionCommand = chain_decorators(
    click.command(),
    config_file_option(),
    checkpoint_path_option(),
)

# pylint: disable=no-value-for-parameter
if __name__ == "__main__":

    @BasePredictionCommand
    def main(config_path, ckpt_path):
        print(f"{config_path=} {ckpt_path=}")

    main()
