
from typing import Any, List
from lightning import LightningModule
from dl_cm.common.datasets.transformations.general_transformation import GeneralRevrsibleTransformation, GeneralTransformationFactory
from dl_cm.common.tasks.base_task import BaseTask

from dl_cm.common.tasks.optimizer import load_optimizer_from_config
from dl_cm.utils.ppattern.batch_operations import merge_multiple_batches, split_into_batches, get_batch_size_from_items_dict
from torchmetrics import MeanMetric
import torch

class MixedVarianceReductionTask(BaseTask):
    
    def __init__(self, task_config: dict):
        #super().__init__(task_config)
        self.variance_reduction_task_config = task_config.get("var_reduction")
        self.transformations_callable : List[GeneralRevrsibleTransformation] = self.init_transformations(self.variance_reduction_task_config.get("transformations"))
        self._var_train_loss_metrics = self.init_loss_metrics()
        self.train_loss_metrics.update(self._var_train_loss_metrics)
        self._var_valid_loss_metrics = self.init_loss_metrics()
        self.valid_loss_metrics.update(self._var_valid_loss_metrics)
    
    @property
    def var_train_loss_metrics(self):
        return self._var_train_loss_metrics
    
    @property
    def var_valid_loss_metrics(self):
        return self._var_valid_loss_metrics
    
    def init_loss_metrics(self):
        metrics_elements = dict()
        metrics_elements[f"total_loss"] = MeanMetric()
        for pred_item_name in self.get_variance_computation_callables():
            metrics_elements[f"{pred_item_name}"] = MeanMetric()
        
        loss_metrics = torch.nn.ModuleDict(metrics_elements)
        return loss_metrics
        
    def init_transformations(self, transformations_config_list):
        return [GeneralTransformationFactory.create(t) for t in transformations_config_list]
        
    def get_variance_computation_callables(self) -> dict:
        raise NotImplementedError
    
    def get_preds_stack_callables(self) -> dict:
        raise NotImplementedError
    
    def check_batch_type(self, batch):
        """
        Default method to diffirentiate between labeled and unlabeled batches
        """
        return "label" in batch.keys()
    
    def unary_step(self, batch: dict)->dict:
        ### example batch = {"images": tensor.ones(10, 3, 128,128)}
        transformed_batches = [t(batch) for t in self.transformations_callable]
        ### example transformed_batches = [ {"images": tensor.ones(10, 3, 128,128)}, {"images": tensor.ones(10, 3, 128,128)} ]
        ### example first element correspond to identity and second element correspond to rot90 
        merged_batches = merge_multiple_batches(transformed_batches)
        ### example merged_baches = {"images": tensor.ones(20, 3, 128,128)}
        ### example first 10 channels correspond to identity and last 10 correspond to rot90
        
        step_output = super().step(merged_batches, compute_loss=False)
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
    
    def step(self, batch):
        is_labeled_batch = self.check_batch_type(batch)
        if is_labeled_batch:
            return super().step(batch)
        else:
            return self.unary_step(batch)
    
    def predict_step(self, batch):
        step_output = super().step(batch, compute_loss=False)
        return step_output
    
    def training_step_end(self, step_output):
        
        if "target" in step_output.keys():
            return super().training_step_end(step_output)
        
        losses_dict = step_output["losses"]
        
        for (loss_name, loss_value) in losses_dict.items():
            self.train_loss_metrics[loss_name].update(loss_value)
            #if self.train_metrics_log_flags.get(loss_name).get("log_on_step"):
            #    self.log(f"train_{loss_name}", loss_value)
        
        return losses_dict["total_loss"]    
    
    def validation_step_end(self, step_output):
        
        if "target" in step_output.keys():
            return super().validation_step_end(step_output)
        
        losses_dict = step_output["losses"]
        
        for (loss_name, loss_value) in losses_dict.items():
            self.valid_loss_metrics[loss_name].update(loss_value)
            #if self.valid_metrics_log_flags.get(loss_name).get("log_on_step"):
            #    self.log(f"valid_{loss_name}", loss_value)
        
        return losses_dict["total_loss"]