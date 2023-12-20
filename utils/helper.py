import PIL
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd


import utils.config as c
import utils.helper as helper

config = c.configuration()


def load_image(path, size, model, upload=False):
    #print(path)
    if upload:
        # Convert the file to an opencv image.
        image = np.asarray(bytearray(path.read()), dtype=np.uint8)
        image = cv2.imdecode(image, 1)
        #image = PIL.Image.open(path)
        #image = np.array(path)
        #print(type(image))
        #print(image.shape)
    else:    
        image = cv2.imread(path)
        #print(type(image))
        #print(image.shape)
    
    image = cv2.resize(image, (size,size))
    if model=='ResNet152V2':
        image = tf.keras.applications.resnet_v2.preprocess_input(image)
    elif model=='VGG19':
        image = tf.keras.applications.vgg19.preprocess_input(image)
    elif model=='EfficientNetB7':
        image = tf.keras.applications.efficientnet.preprocess_input(image)
    elif model=='EfficientNetV2S':
        image = tf.keras.applications.efficientnet_v2.preprocess_input(image)
    elif model=='ConvNeXtBase':
        pass
    elif model=='ResAttUnet':
        config.DEPTH=1
        config.SIZE=256
        if config.DEPTH==1:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image/255.
    #elif model=='vit_b16':
    #  image = vit.preprocess_inputs(image)
    #if config.DEPTH==1:
    #  image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #image = image/255.
    return image

def make_df(predictions, class_labels=['normal', 'benign', 'malignant']):
    df = pd.DataFrame({'Class': class_labels, 'Prediction in %': predictions[0]})
    df['Prediction in %'] = pd.to_numeric(df['Prediction in %'], errors='coerce')
    df = df.sort_values(by='Prediction in %', ascending=False)
    df['Prediction in %'] = df['Prediction in %'].apply(lambda x: round(x*100,1))
    return df

