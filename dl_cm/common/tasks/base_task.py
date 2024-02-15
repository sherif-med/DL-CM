import pytorch_lightning as pl
from dl_cm.common.tasks.metrics import init_metric_collection_from_config
from dl_cm.common.tasks.criterion import init_loss_metrics_dict_from_config, load_critireon_from_config

class BaseTask(pl.LightningModule):
    def __init__(self, task_config:dict):
        super().__init__()
        self.task_config: dict = task_config
        self.save_hyperparameters(task_config)
        # Set the task name to the name of the current class
        self.hparams["task_name"] = type(self).__name__
        
        self.criterion = load_critireon_from_config(self.task_config.get("critireon"))
        self.train_loss_metrics, self.train_metrics_log_flags = init_loss_metrics_dict_from_config(self.task_config.get("critireon"))
        self.valid_loss_metrics, self.valid_metrics_log_flags = init_loss_metrics_dict_from_config(self.task_config.get("critireon"))
        self.train_metrics, self.valid_metrics = init_metric_collection_from_config(self.task_config.get("metrics"))        
        # Call the post initialization method
        self.post_init()

    def post_init(self):
        """
        This method is called at the end of the BaseTask __init__ method.
        """
        pass

    def info(self):
        """
        Returns the task description defined in the child class.
        """
        # Check if the child class has a DESCRIPTION attribute
        if hasattr(self, 'DESCRIPTION'):
            return self.DESCRIPTION
        else:
            # Return a default message or raise an error if DESCRIPTION is not defined
            return "No description provided."

