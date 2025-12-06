from pathlib import Path
from skimage.io import imsave
from pytorch_lightning.callbacks import BasePredictionWriter, Callback
from dl_cm.common.trainer.callbacks import baseCallback
import os


class ImagesPredictionWriter(BasePredictionWriter, baseCallback):

    #@rank_zero_only
    def __init__(
        self,
        output_dir,
        predicted_map_key,
        write_interval="batch",
    ):
        super().__init__(write_interval)
        self.output_dir = output_dir
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)
        self.predicted_map_key = predicted_map_key

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        # TODO: Treat more edge cases (this is very weak impl)
        preds, filenames = prediction["predictions"][self.predicted_map_key], batch["inputs"]["id"]
        preds = preds.cpu().numpy().astype('uint8') * 255

        for prediction, filename in zip(preds, filenames):
            output_file = Path(self.output_dir, os.path.basename(filename)).with_suffix(".png")
            imsave(output_file, prediction, check_contrast=False)
    
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not self.interval.on_batch:
            return

        batch_indices = trainer.predict_loop.current_batch_indices
        self.write_on_batch_end(
            trainer, pl_module, outputs, batch_indices, batch, batch_idx, dataloader_idx
            )
        
class PostPredictionCallback(BasePredictionWriter):
    
    def __init__(self, output_dir, write_interval, preds_items_callback):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.preds_items_callback = preds_items_callback
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)
    
    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        """"""        
        for c_pred_key, c_pred_callable in self.preds_items_callback.items():
            c_pred = prediction["preds"][c_pred_key]
            c_pred_callable(self.output_dir, c_pred, batch)
