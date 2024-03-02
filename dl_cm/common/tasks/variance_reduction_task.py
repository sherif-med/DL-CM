
from typing import Any, List
from pytorch_lightning import LightningModule, Trainer
from dl_cm.common.datasets.transformations.general_transformation import GeneralRevrsibleTransformation, GeneralTransformationFactory
from dl_cm.common.tasks.base_task import BaseTask

from dl_cm.common.tasks.metrics import init_metric_collection_from_config
from dl_cm.common.tasks.criterion import init_loss_metrics_dict_from_config, load_critireon_from_config
from dl_cm.common.tasks.optimizer import load_optimizer_from_config
from dl_cm.utils.ppattern.batch_operations import merge_multiple_batches, split_into_batches, get_batch_size_from_items_dict
from torchmetrics import MeanMetric
import torch

class VarianceReductionTask(LightningModule):
    
    def __init__(self, wraped_task: BaseTask, variance_reduction_task_config: dict):
        super().__init__()
        self.wraped_task : BaseTask= wraped_task
        self.variance_reduction_task_config=variance_reduction_task_config
        self.transformations_callable : List[GeneralRevrsibleTransformation] = self.init_transformations(self.variance_reduction_task_config.get("transformations"))
        self.train_loss_metrics = self.init_loss_metrics()
        self.valid_loss_metrics = self.init_loss_metrics()
    
    def init_loss_metrics(self):        
        metrics_elements = dict()
        metrics_elements[f"total_loss"] = MeanMetric()
        for pred_item_name in self.get_variance_computation_callables():
            metrics_elements[f"{pred_item_name}"] = MeanMetric()
        
        loss_metrics = torch.nn.ModuleDict(metrics_elements)
        return loss_metrics
        
    
    def init_transformations(self, transformations_config_list):
        return [GeneralTransformationFactory.create(t) for t in transformations_config_list]
    
    def configure_optimizers(self):
        return load_optimizer_from_config(self.wraped_task.model.parameters(), self.variance_reduction_task_config.get("optimizer"))
    
    def get_variance_computation_callables(self) -> dict:
        raise NotImplementedError
    
    def get_preds_stack_callables(self) -> dict:
        raise NotImplementedError
    
    def step(self, batch: dict)->dict:
        ### example batch = {"images": tensor.ones(10, 3, 128,128)}
        transformed_batches = [t(batch) for t in self.transformations_callable]
        ### example transformed_batches = [ {"images": tensor.ones(10, 3, 128,128)}, {"images": tensor.ones(10, 3, 128,128)} ]
        ### example first element correspond to identity and second element correspond to rot90 
        merged_batches = merge_multiple_batches(transformed_batches)
        ### example merged_baches = {"images": tensor.ones(20, 3, 128,128)}
        ### example first 10 channels correspond to identity and last 10 correspond to rot90
        
        step_output = self.wraped_task.step(merged_batches, compute_loss=False)
        ### example step_output = {"loss": 0.2, "preds":{"seg_map": tensor.ones(20, 1, 128,128), "proba_map": tensor.ones(20, 1, 128,128)}}
        ### example first 10 channels in seg_map or proba_map correspond to identity and last 10 correspond to rot90
        
        split_prediction : List[dict] = split_into_batches(step_output["preds"], get_batch_size_from_items_dict(batch))
        ### example split_prediction = [{"seg_map": torch.tensor(10, 1,128,128), "proba_map": same}, {"seg_map": torch.tensor(10, 1,128,128), "proba_map": same}]
        ### first item dict in split_prediction correspond to identity prediction result and second item dict correspond to rot90 predicition results
        
        reverse_transformed_prediction = []
        ### example reverse_transformed_prediction = will have the same shape as split_prediction
        ### It will contain items but after performing reverse transfromation
        
        for c_prediction, c_respective_transform in zip(split_prediction, self.transformations_callable):
            reverse_transformed_prediction.append(
                c_respective_transform(c_prediction, reverse=True)
            )
            
        losses_dict = {}
        for pred_item_name, c_var_fn in self.get_variance_computation_callables().items():
            losses_dict[pred_item_name] = c_var_fn([pred[pred_item_name] for pred in reverse_transformed_prediction])
        
        preds_dict = {}
        for pred_item_name, c_pred_fn in self.get_preds_stack_callables().items():
            preds_dict[pred_item_name] = c_pred_fn([pred[pred_item_name] for pred in reverse_transformed_prediction])
        
        losses_dict["total_loss"]=sum(losses_dict.values())
        
        output = {
            "loss": losses_dict["total_loss"],
            "losses": losses_dict,
            "preds": preds_dict,
        }
        return output
    
    def training_step(self, batch):
        step_output = self.step(batch)
        self.training_step_end(step_output)
        return step_output
    
    def validation_step(self, batch):
        step_output = self.step(batch)
        self.validation_step_end(step_output)
        return step_output
    
    def training_step_end(self, step_output):        
        losses_dict = step_output["losses"]        
        for (loss_name, loss_value) in losses_dict.items():
            self.train_loss_metrics[loss_name].update(loss_value)
        return losses_dict["total_loss"]    
    
    def validation_step_end(self, step_output):
        losses_dict = step_output["losses"]
        for (loss_name, loss_value) in losses_dict.items():
            self.valid_loss_metrics[loss_name].update(loss_value)
        return losses_dict["total_loss"]
        
    def on_validation_epoch_end(self):
        for c_loss_name, c_loss_metric in self.valid_loss_metrics.items():
            c_loss_value = c_loss_metric.compute()
            self.log(f"valid_{c_loss_name}", c_loss_value)
            c_loss_metric.reset()
    
    def on_train_epoch_end(self):
        for c_loss_name, c_loss_metric in self.train_loss_metrics.items():
            c_loss_value = c_loss_metric.compute()
            self.log(f"train_{c_loss_name}", c_loss_value)
            c_loss_metric.reset()
        