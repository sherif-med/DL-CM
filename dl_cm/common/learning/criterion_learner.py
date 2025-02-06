from dl_cm.common.learning.base_learner import BaseLearner
from dl_cm.utils.ppattern.data_validation import validationMixin
from dl_cm.common.tasks.criterion import CritireonFactory
import pydantic as pd
from torchmetrics import MetricCollection
from dl_cm.common.tasks.criterion import BaseLoss
from dl_cm.common.tasks.base_task import StepOutputDict

class CriterionLearner(BaseLearner, validationMixin):

    def config_schema(cls)-> pd.BaseModel:        
        class ValidConfig(pd.BaseModel):
            critireon: dict
        return ValidConfig

    def __init__(self, learner_config:dict):
        validationMixin.__init__(self, learner_config)
        BaseLearner.__init__(self, learner_config)
        self.criterion : BaseLoss = CritireonFactory.create(self.config.get("critireon"))
        self.train_loss_metrics : MetricCollection
        self.valid_loss_metrics : MetricCollection
        self.init_metrics()
    
    def init_metrics(self):
        # Loss metrics initialization (train)
        train_loss_metrics : MetricCollection = \
              self.criterion.as_metric_collection().clone(prefix="train_")
        if self.train_loss_metrics is None:
            self.train_loss_metrics = train_loss_metrics
        else:
            self.train_loss_metrics.add_metrics(train_loss_metrics)
        # Loss metrics initialization (valid)
        valid_loss_metrics : MetricCollection = \
            self.train_loss_metrics.clone(prefix="valid_")
        if self.valid_loss_metrics is None:
            self.valid_loss_metrics = valid_loss_metrics
        else:
            self.valid_loss_metrics.add_metrics(valid_loss_metrics)
    
    def on_train_batch_end(self, step_output:StepOutputDict):
        losses_dict = step_output["loss"]        
        for (loss_name, loss_value) in losses_dict.items():
            self.train_loss_metrics[loss_name].update(loss_value)
        
    
    def on_validation_batch_end(self, step_output:StepOutputDict):
        losses_dict = step_output["loss"]        
        for (loss_name, loss_value) in losses_dict.items():
            self.valid_loss_metrics[loss_name].update(loss_value)        
    
    def on_train_epoch_end(self):
        self.train_loss_metrics.compute()
    
    def on_validation_epoch_end(self):
        self.valid_loss_metrics.compute()
    
    def on_train_epoch_start(self):
        self.train_loss_metrics.reset()
        
    def on_validation_epoch_start(self):
        self.valid_loss_metrics.reset()
    