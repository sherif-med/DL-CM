
from dl_cm.scripts import BaseTrainingCommand
from dl_cm.common.datasets.datamodule import BaseDataModule
from dl_cm.common.trainers.base_trainer import TrainerFactory
from dl_cm.common.tasks import TasksFactory
from dl_cm.utils.config_validation import validate_config
from dl_cm.config_loaders import open_config_file
from pytorch_lightning import seed_everything

@BaseTrainingCommand
def train(config_path, ckpt_path, seed):
    if seed!=-1:
        seed_everything(seed, workers=True)        
    
    validate_config(config_path, True)    
    config : dict = open_config_file(config_path)    
    loaded_datamodule = BaseDataModule(config.get("datainfo"))
    loaded_trainer = TrainerFactory.create(config.get("trainer"))
    loaded_task = TasksFactory.create(config.get("task"))
    loaded_task.to("cuda")
    loaded_trainer.fit(loaded_task, datamodule=loaded_datamodule, ckpt_path=ckpt_path)
    

if __name__ == "__main__":
    train()