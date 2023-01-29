from torchmetrics import BLEUScore
import evaluate
from typing import List

class ImageCaptionMetrics:
  """Collect all the metrics in one class"""
  meteor = None
  
  @classmethod
  def bleu_score(cls, preds : str, targets : List[str], n_gram=2):
    bleu = BLEUScore(n_gram=n_gram)
    return bleu([preds], [targets])
  
  @classmethod
  def meteor_score(cls, preds : str, targets : List[str]):
    
    if cls.meteor is None:
      cls.meteor = evaluate.load("meteor")
      
    # HuggingFace implementation of METEOR metric expects a list of reference words
    # and a list of predictions words.
    target_list = [t.split() for t in targets]
    return cls.meteor.compute(predictions=[preds.split()], references=target_list)
