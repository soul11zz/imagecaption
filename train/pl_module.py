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
        self.is_semantic_validation = (metric == "semantic")
        
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
      self.log_dict({"train_loss": loss})
      return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):

        input_ids = batch["input_ids"]
        pixel_values = batch["pixel_values"]

        outputs = self.model(input_ids=input_ids,
                        pixel_values=pixel_values,
                        labels=input_ids)
        
        loss = outputs.loss
        
        if self.is_semantic_validation:
            semanic_sim = 0
            for b, p in zip(input_ids, pixel_values):
                test_batch = {"input_ids": b.unsqueeze(0), "pixel_values": p.unsqueeze(0)}
                semanic_sim += (1 - self.compute_metric(test_batch, None))
            semanic_sim /= input_ids.shape[0]
            
            loss = (outputs.loss, semanic_sim,)    
        return loss

    def validation_epoch_end(self, outputs: Union[float, List[float]]) -> None:
        out_loss = outputs
        
        if self.is_semantic_validation:
            out_loss, out_sem = zip(*outputs)
            avg_sem = sum(out_sem) / len(out_sem)
            self.log_dict({"semantic_distance": avg_sem}, sync_dist=True)
            
        avg_loss = sum(out_loss) / len(out_loss)
        
        try:
            perplexity = torch.exp(avg_loss)
        except OverflowError:
            perplexity = float("inf")
        
        self.log_dict({"val_loss": avg_loss, "perplexity": perplexity}, sync_dist=True)
        return avg_loss

    def compute_metric(self, batch, batch_idx):
        labels = self.processor.decode(batch["input_ids"][0], skip_special_tokens=True)
        pixel_values = batch["pixel_values"]
        
        outputs = self.model.generate(pixel_values = pixel_values,
                                max_length=150,
                                early_stopping=True,
                            )

        preds = self.processor.decode(outputs[0], skip_special_tokens=True)
        return self.metric(preds, [labels])
    
    def test_step(self, batch, batch_idx):
        return self.compute_metric(batch, batch_idx)
    
    def test_epoch_end(self, test_scores):
        score = sum(test_scores) / len(test_scores)
        logging.info(f"Average {self.metric_name.upper()}: {score}")
        self.log_dict({self.metric_name: score})
      
    def configure_optimizers(self):
        # TODO add scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        monitor_loss = "val_loss" if not self.is_semantic_validation else "semantic_distance"
        
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
        scheduler = {"scheduler": lr_scheduler, "monitor": monitor_loss, "interval": "epoch"}
        
        return [optimizer], [scheduler]
      