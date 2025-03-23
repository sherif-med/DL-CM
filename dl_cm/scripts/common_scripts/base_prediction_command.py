import click

from . import (
    chain_decorators,
    checkpoint_path_option,
    input_folder_option,
    output_folder_option,
    pytorch_accelerator_option,
)

BasePredictionCommand = chain_decorators(
    click.command(),
    input_folder_option(),
    output_folder_option(),
    checkpoint_path_option(),
    pytorch_accelerator_option(),
)

# pylint: disable=no-value-for-parameter
if __name__ == "__main__":

    @BasePredictionCommand
    def main(input_folder, output_folder, ckpt_path, device):
        print(f"{input_folder=} {output_folder=} {ckpt_path=} {device=}")

    main()
