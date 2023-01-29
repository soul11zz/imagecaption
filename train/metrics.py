from torchmetrics import BLEUScore
import evaluate
from typing import List

class ImageCaptionMetrics:
  """Collect all the metrics in one class"""
  meteor = None
  bleu = None
  
  @classmethod
  def bleu_score(cls, preds : str, targets : List[str], n_gram=2):
    if cls.bleu is None:
      cls.bleu = BLEUScore(n_gram=n_gram)
    return cls.bleu([preds], [targets]).item()
  
  @classmethod
  def meteor_score(cls, preds : str, targets : List[str]):
    
    if cls.meteor is None:
      cls.meteor = evaluate.load("meteor")
      
    # HuggingFace implementation of METEOR metric expects a list of reference words
    # and a list of predictions words.
    
    return cls.meteor.compute(predictions=[preds], references=targets)["meteor"]
