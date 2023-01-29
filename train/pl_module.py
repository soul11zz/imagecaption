import torch
import pytorch_lightning as pl
from metrics import ImageCaptionMetrics

import logging
logging.basicConfig(format='%(asctime)s  %(levelname)-10s %(message)s', datefmt="%Y-%m-%d-%H-%M-%S", level=logging.INFO)


class ImageCaptioningModule(pl.LightningModule):
    def __init__(self, processor, model, train_dataloader, val_dataloader, test_dataloader = None, learning_rate=1e-2, batch_size=2):
        super().__init__()
        self.model = model
        self.processor = processor
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.test_loader = test_dataloader
        self.lr = learning_rate
        self.batch_size = batch_size

    def training_step(self, batch, batch_idx):
      input_ids = batch["input_ids"]
      pixel_values = batch["pixel_values"]

      outputs = self.model(input_ids=input_ids,
                      pixel_values=pixel_values,
                      labels=input_ids)
    
      loss = outputs.loss
      self.log_dict({"train_loss": loss}, sync_dist=True)
      return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):

      input_ids = batch["input_ids"]
      pixel_values = batch["pixel_values"]

      outputs = self.model(input_ids=input_ids,
                      pixel_values=pixel_values,
                      labels=input_ids)
    
      loss = outputs.loss
      try:
        perplexity = torch.exp(loss)
      except OverflowError:
        perplexity = float("inf")
        
      self.log_dict({"val_loss": loss, "perplexity": perplexity}, sync_dist=True)
      return loss

    def test_step(self, batch, batch_idx):
      labels = self.processor.decode(batch["input_ids"][0], skip_special_tokens=True)
      pixel_values = batch["pixel_values"]
      
      outputs = self.model.generate(pixel_values = pixel_values,
                                max_length=50,
                                early_stopping=True,
                            )

      preds = self.processor.decode(outputs[0], skip_special_tokens=True)
      return ImageCaptionMetrics.meteor_score(preds, [labels])
    
    def test_epoch_end(self, test_scores):
      scores = [m["meteor"] for m in test_scores]
      logging.info(f"Average METEOR: {sum(scores) / len(scores)}")
      self.log_dict({"meteor": sum(scores) / len(scores)})
      
    def configure_optimizers(self):
        # TODO add scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    
        return optimizer

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
      
    def test_dataloader(self):
        return self.test_loader
      