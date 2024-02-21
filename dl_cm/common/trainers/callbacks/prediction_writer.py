from pathlib import Path
from skimage.io import imsave
from pytorch_lightning.callbacks import BasePredictionWriter
import numpy as np
import os


class ImagesPredictionWriter(BasePredictionWriter):

    #@rank_zero_only
    def __init__(
        self,
        output_dir,
        write_interval,
        predicted_map_key
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
        preds, filenames = prediction["preds"][self.predicted_map_key], batch["id"]
        preds = preds.cpu().numpy().astype('uint8')

        for prediction, filename in zip(preds, filenames):
            output_file = Path(self.output_dir, os.path.basename(filename)).with_suffix(".png")
            imsave(output_file, prediction[0], check_contrast=False)
    
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not self.interval.on_batch:
            return

        batch_indices = trainer.predict_loop.current_batch_indices
        self.write_on_batch_end(
            trainer, pl_module, outputs, batch_indices, batch, batch_idx, dataloader_idx
            )
        
