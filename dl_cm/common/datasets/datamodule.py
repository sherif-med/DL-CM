import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS


class PartialDataModule(pl.LightningDataModule):
    
    def __init__(self, train_partial=None, val_parital=None, test_partial=None, prediction_partial=None):
        super().__init__()
        self.train_partial = train_partial
        self.val_parital = val_parital
        self.test_partial = test_partial
        self.prediction_partial = prediction_partial
    
    
    def train_dataloader(self):
        return self.train_partial()
    
    def val_dataloader(self):
        return self.val_parital()
    
    def test_dataloader(self):
        return self.test_partial()
    
    def predict_dataloader(self):
        return self.prediction_partial()