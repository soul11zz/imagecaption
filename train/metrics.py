from torchmetrics import BLEUScore
import evaluate
from typing import List

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
class ImageCaptionMetrics:
  """Collect all the metrics in one class"""
  meteor = None
  bleu = None
  semantic_model = None
  
  @classmethod
  def bleu_score(cls, preds : str, targets : List[str], n_gram=4):
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

  @classmethod
  def semantic_score(cls, preds : str, targets : List[str]):
    if cls.semantic_model is None:
      cls.semantic_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
      
    encodings_preds = cls.semantic_model.encode([preds], convert_to_tensor=True)
    encodings_expected = cls.semantic_model.encode(targets, convert_to_tensor=True)
    return cos_sim(encodings_preds, encodings_expected).item()