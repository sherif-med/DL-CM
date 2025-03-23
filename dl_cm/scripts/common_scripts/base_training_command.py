import click

from . import (
    chain_decorators,
    checkpoint_path_option,
    config_file_option,
    seeding_option,
)

BaseTrainingCommand = chain_decorators(
    click.command(), config_file_option(), checkpoint_path_option(), seeding_option()
)


# pylint: disable=no-value-for-parameter
def open_config_file(conf_file_path: str) -> dict:
    import yaml

    with open(conf_file_path, "r") as f:
        config = yaml.safe_load(f)
        return config


if __name__ == "__main__":

    @BaseTrainingCommand
    def main(config_path, ckpt_path, seed):
        print(f"{config_path=}")
        if config_path:
            config = open_config_file(config_path)
            print(config)

    main()
