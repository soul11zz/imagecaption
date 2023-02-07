from typing import List, Union
import torch
import pytorch_lightning as pl
from metrics import ImageCaptionMetrics

import logging
logging.basicConfig(format='%(asctime)s  %(levelname)-10s %(message)s', datefmt="%Y-%m-%d-%H-%M-%S", level=logging.INFO)


class ImageCaptioningModule(pl.LightningModule):
    def __init__(self, processor, model, learning_rate=1e-2, metric="meteor"):
        super().__init__()
        self.model = model
        self.processor = processor
        self.lr = learning_rate
        
        # Resolve the metric
        metric_score = f"{metric}_score"
        assert hasattr(ImageCaptionMetrics, metric_score), f"Metric {metric} not found in ImageCaptionMetrics"
        
        self.metric = getattr(ImageCaptionMetrics, metric_score)
        self.metric_name = metric
        
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

      pred_outputs = self.model.generate(pixel_values = pixel_values[0].unsqueeze(0),
                                max_length=50,
                                early_stopping=True,
                            )

      preds = self.processor.decode(pred_outputs[0], skip_special_tokens=True)
      loss = outputs.loss
      return loss

    def validation_epoch_end(self, outputs: Union[float, List[float]]) -> None:
      avg_loss = sum(outputs) / len(outputs)
      self.log_dict({"val_loss": avg_loss}, sync_dist=True)
      
      try:
        perplexity = torch.exp(avg_loss)
      except OverflowError:
        perplexity = float("inf")
      
      self.log_dict({"val_loss": avg_loss, "perplexity": perplexity}, sync_dist=True)
      return avg_loss
     
    def test_step(self, batch, batch_idx):
      labels = self.processor.decode(batch["input_ids"][0], skip_special_tokens=True)
      pixel_values = batch["pixel_values"]
      
      outputs = self.model.generate(pixel_values = pixel_values,
                                max_length=50,
                                early_stopping=True,
                            )

      preds = self.processor.decode(outputs[0], skip_special_tokens=True)
      return self.metric(preds, [labels])
    
    def test_epoch_end(self, test_scores):
      score = sum(test_scores) / len(test_scores)
      logging.info(f"Average {self.metric_name.upper()}: {score}")
      self.log_dict({self.metric_name: score})
      
    def configure_optimizers(self):
        # TODO add scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=2)
        scheduler = {"scheduler": lr_scheduler, "monitor": "val_loss"}
    
        return [optimizer], scheduler
      