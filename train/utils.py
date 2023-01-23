from PIL import Image
import numpy as np

def denormalize(batch):

  MEAN = np.array([123.675, 116.280, 103.530]) / 255
  STD = np.array([58.395, 57.120, 57.375]) / 255

  unnormalized_image = (batch["pixel_values"][0].numpy() * np.array(STD)[:, None, None]) + np.array(MEAN)[:, None, None]
  unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
  unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
  return Image.fromarray(unnormalized_image)  