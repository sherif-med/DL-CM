import pytorch_lightning as pl
from dl_cm.common.tasks.metrics import init_metric_collection_from_config
from dl_cm.common.tasks.criterion import init_loss_metrics_dict_from_config, load_critireon_from_config
from dl_cm.common.tasks.optimizer import load_optimizer_from_config

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
    
    def step(self, batch) -> dict:
        raise NotImplementedError
    
    def training_step_end(self, step_output):
        
        losses_dict = step_output["losses"]
        preds = step_output["preds"]
        targets = step_output["target"]
        
        for (loss_name, loss_value) in losses_dict.items():
            self.train_loss_metrics[loss_name].update(loss_value)
            if self.train_metrics_log_flags.get(loss_name).get("log_on_step"):
                self.log(f"train_{loss_name}", loss_value)
        
        metrics_output = self.train_metrics(preds=preds, target=targets)
        
        self.log_dict(metrics_output)
        return losses_dict["total_loss"]    
    
    def validation_step_end(self, step_output):
        
        losses_dict = step_output["losses"]
        preds = step_output["preds"]
        targets = step_output["target"]
        
        for (loss_name, loss_value) in losses_dict.items():
            self.valid_loss_metrics[loss_name].update(loss_value)
            if self.valid_metrics_log_flags.get(loss_name).get("log_on_step"):
                self.log(f"valid_{loss_name}", loss_value)
        
        metrics_output = self.valid_metrics(preds=preds, target=targets)
        
        self.log_dict(metrics_output)
        return losses_dict["total_loss"]
        
    def on_validation_epoch_end(self):
        for c_loss_name, c_loss_metric in self.valid_loss_metrics.items():
            if self.valid_metrics_log_flags.get(c_loss_name).get("log_on_epoch"):
                c_loss_value = c_loss_metric.compute()
                self.log(f"valid_{c_loss_name}", c_loss_value)
            c_loss_metric.reset()
            
        metrics_epoch = self.valid_metrics.compute()
        self.log_dict(metrics_epoch)        
        self.valid_metrics.reset()
    
    def on_train_epoch_end(self):
        for c_loss_name, c_loss_metric in self.train_loss_metrics.items():
            if self.train_metrics_log_flags.get(c_loss_name).get("log_on_epoch"):
                c_loss_value = c_loss_metric.compute()
                self.log(f"train_{c_loss_name}", c_loss_value)
            c_loss_metric.reset()
            
        metrics_epoch = self.train_metrics.compute()
        self.log_dict(metrics_epoch)        
        self.train_metrics.reset()
    
    def configure_optimizers(self):
        return load_optimizer_from_config(self.model.parameters(), self.task_config.get("optimizer"))
