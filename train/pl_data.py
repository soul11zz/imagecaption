import pytorch_lighting as pl
from dataset import ImageCaptioningDataset
import os
from torch.utils.data import DataLoader

class ImageCaptionDataModule(pl.DataModule):
    def __init__(self):
        super().__init__(dataset_path, processor, **kwargs)
        
        self.dataset_path = dataset_path
        self.processor = processor
        self.prepare_data_per_node = True
        
        self.save_hyperparameters(kwargs)
        self.batch_size = self.hparams.get("batch_size", 1)
    
    def prepare_data(self):
        load_dataset(dataset_path, split="train")
    
    def setup(self, stage=None):

        processor = self.processor
        
        if stage != "test":
            dt_train = load_dataset(dataset_path, split="train")
            self.train_dataset = ImageCaptioningDataset(dt_train, processor)
            dt_val = load_dataset(self.dataset_path, split="validation")
            self.val_dataset = ImageCaptioningDataset(dt_val, processor)                
        else:
            dt_test = load_dataset(self.dataset_path, split="test")
            self.test_dataset = ImageCaptioningDataset(dt_test, processor)
            
        self.num_workers = os.cpu_count() if os.name != "nt" else 0
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers)
        