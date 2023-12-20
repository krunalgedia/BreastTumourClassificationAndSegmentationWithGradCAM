import os


class Paths:
  def __init__(self):
    self.DETECTION_MODEL = 'Detection_model'
    self.SEGMENTATION_MODEL   = 'Segmentation_model'
    self.DEFAULT_DETECTION_IMAGE = os.path.join('Dataset_BUSI_with_GT','benign','benign (1).png')
    self.DEFAULT_DETECTION_GRADCAM_IMAGE = os.path.join('Dataset_BUSI_with_GT','benign','benign (1).png')
     



if __name__=='__main__':
  pass  

paths = Paths()
