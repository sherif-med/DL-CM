import click
from enum import Enum


def output_folder_option(short_name="-o", long_name="--output_folder")->click.Option:
    return click.option(short_name, long_name, type=click.Path(file_okay=False), help="Output folder for prediction!")

def input_folder_option(short_name="-i", long_name="--input_folder")->click.Option:
    return click.option(short_name, long_name, type=click.Path(exists=True, file_okay=False) ,help="Input folder to run inference!")


class PytorchAcceleratorsEnum(Enum):
    CPU = "cpu"
    GPU = "cuda"

def pytorch_accelerator_option(short_name="-d", long_name="--device")->click.Option:
    """
    Returns:
        click.Option: Device enumerator option
    """
    import torch
    default_accelerator = PytorchAcceleratorsEnum.CPU
    if torch.cuda.is_available():
        default_accelerator = PytorchAcceleratorsEnum.GPU
    return click.option(short_name, long_name, type=click.Choice(PytorchAcceleratorsEnum.__members__), 
              callback=lambda c, p, v: getattr(PytorchAcceleratorsEnum, v).value if v else None, default=default_accelerator.name,
              help="device to use pytorch operations!")


def checkpoint_path_option(short_name="-c", long_name="--ckpt_path")->click.Option:
    return click.option(short_name, long_name, type=click.Path(exists=True, dir_okay=False),
                        help="Checkpoint file path!")


def config_file_option(short_name="-k", long_name="--config_path")->click.Option:
    return click.option(short_name, long_name, type=click.Path(exists=True, dir_okay=False),
                        help="Config file path!")


from collections.abc import Iterable
def chain_decorators(*args):
    """
    Returns a dinal chained decorator from input decorators
    """
    # if fucntions are provided as an iterable
    if len(args)==1 and isinstance(args[0], Iterable):
        args = list(args[0])
    
    assert len(args)!=0, "Input decorators are empty!"
    assert all([callable(el) for el in args]), "Some arguments are not callable!"
    
    def decorator(func):
        for deco in reversed(args):
            func = deco(func)
        return func
    return decorator
    