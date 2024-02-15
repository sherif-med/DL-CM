import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS


class PartialDataModule(pl.LightningDataModule):
    
    def __init__(self, train=None, valid=None, test=None, prediction=None):
        super().__init__()
        self.train_partial = train
        self.val_parital = valid
        self.test_partial = test
        self.prediction_partial = prediction
    
    
    def train_dataloader(self):
        return self.train_partial()
    
    def val_dataloader(self):
        return self.val_parital()
    
    def test_dataloader(self):
        return self.test_partial()
    
    def predict_dataloader(self):
        return self.prediction_partial()