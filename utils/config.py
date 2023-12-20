import os
import tensorflow as tf


class configuration():
  def __init__(self):
    self.set_image()
    self.set_model()
    self.set_dir()
    self.get_model_config()

  def set_image(self):
    self.BATCH_SIZE = 16
    self.TEST_SIZE  = 0.2
    self.SEED       = 24
    self.SHUFFLE    = True
    self.AUTOTUNE   = tf.data.AUTOTUNE
    self.HEIGHT     = 256
    self.WIDTH      = 256
    self.DEPTH      = 3
    self._size      = 256

  @property
  def SIZE(self):
    if (self.HEIGHT != self.WIDTH):
      raise ValueError("size, height width mis match in config")
    return(self._size)

  @SIZE.setter
  def SIZE(self,value):
    self._size  = value
    self.WIDTH  = value
    self.HEIGHT = value
    return(self._size)

  def set_model(self):
    self.LR = 0.001
    self.NUM_EPOCHS=15

  def set_dir(self):
    self.DATA_DIR  = 'Dataset_BUSI_with_GT/'
    self.BENIGN    = 'benign'
    self.MALIGNANT = 'malignant'
    self.NORMAL    = 'normal'

    self.BENIGN_DIR    = os.path.join(self.DATA_DIR,self.BENIGN)
    self.MALIGNANT_DIR = os.path.join(self.DATA_DIR,self.MALIGNANT)
    self.NORMAL_DIR    = os.path.join(self.DATA_DIR,self.NORMAL)


  def get_model_config(self):
    self.DETECTION_MODEL = 'EfficientNetB7'
    self.DETECTION_MODEL_GRADCAM_LAYER2 = 'top_activation' 
    self.DETECTION_MODEL_GRADCAM_LAYER1 = 'efficientnetb7'
    self.SEGMENTATION_MODEL = 'ResAttUnet'
    self.SEGMENTATION_MODEL_GRADCAM_LAYER1 = 'att2_wgt'

