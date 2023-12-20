import os
import sys
import numpy as np
import pandas as pd
import cv2
import PIL
import tensorflow as tf
from tf_explain.core.grad_cam import GradCAM
from tensorflow.keras import backend as K
import utils.config as c
import utils.paths as p
import utils.metrics as metrics
import utils.helper as helper
import streamlit as st

paths = p.Paths()
config = c.configuration()

def get_DETECTION_col2(path, size, upload=False):
    test_img = helper.load_image(path=path, size = size, model = config.DETECTION_MODEL, upload = upload)
    test_img = np.expand_dims(test_img, axis=0)
    pred = model.predict(test_img)

    target_class = np.argmax(pred,axis=1)[0]
    explainer = GradCAM()
    grad_cam_img = explainer.explain((test_img,test_img[0,...]),
                       model.get_layer(config.DETECTION_MODEL_GRADCAM_LAYER1),
                       layer_name = config.DETECTION_MODEL_GRADCAM_LAYER2,
                       class_index=target_class,
                       image_weight=0.70)
    df = helper.make_df(pred)

    condition = df.iloc[0,0]
    st.header(':violet[AI Model prediction:] '+':red['+condition+']')
    st.table(df)
    st.header(':red[AI model focus area]')
    st.image(grad_cam_img,
             caption='Grad CAM image',
             use_column_width=True
            )


def get_SEGMENTATION_col2(path, size, upload=False):
    test_img = helper.load_image(path=path, size = size, model = config.SEGMENTATION_MODEL, upload=upload)
    test_img = np.expand_dims(test_img, axis=0)
    pred = model.predict(test_img)
    if thres:
        pred = pred[0] > thres
        pred = pred.astype(int)
    explainer = GradCAM()
    grad_cam_img = explainer.explain((test_img,test_img[0,...]),
                       model,
                       layer_name = config.SEGMENTATION_MODEL_GRADCAM_LAYER1,
                       class_index=0,
                       image_weight=0.70)
    st.header(':violet[AI Model Prediction]')
    st.image(pred,
             caption='Segmentation Scan',
             use_column_width=True
            )
    st.header(':red[AI model focus area]')
    st.image(grad_cam_img,
             caption='Grad CAM image',
             use_column_width=True
            )

st.set_page_config(
    page_title = 'AI Brain tumour Detection and Segmentation',
    layout = 'wide',
    initial_sidebar_state = 'expanded'
    )
st.write("<style>div.Widget.stTitle {margin-top: -80px;}</style>", unsafe_allow_html=True)
st.title('AI Brain tumour Detection and Segmentation')
st.sidebar.header('Task & Config')

model_type = st.sidebar.radio("Select Task", ['Detection', 'Segmentation'])
thres_type = st.sidebar.radio("Sharpness Threshold (Seg. Model)", ['False', 'True'])

thres = None
if thres_type == 'True':
    thres = st.sidebar.slider('Sharpness', min_value=0.0, max_value=1.0, value=0.15)

if model_type == 'Detection':
    model_path = paths.DETECTION_MODEL
    custom_objects = {'recall_c0':metrics.recall_c0,'recall_c1':metrics.recall_c1,'recall_c2':metrics.recall_c2, 
                      'precision_c0':metrics.precision_c0,'precision_c1':metrics.precision_c1,'precision_c2':metrics.precision_c2}                 
    try:
        model = tf.keras.models.load_model(model_path,custom_objects=custom_objects, safe_mode=False)
    except Exception as ex:
        st.error(f'Unable to load the pretrained model with weight at {model_path}')
        st.error(ex)
  
elif model_type == 'Segmentation':
    model_path = paths.SEGMENTATION_MODEL

    try:
        model = tf.keras.models.load_model(model_path, safe_mode=False)
    except Exception as ex:
        st.error(f'Unable to load the pretrained model with weight at {model_path}')
        st.error(ex)

source_img = None

if model_type == 'Detection':
  col1, col2 = st.columns(2,gap='large')
  source_img = st.sidebar.file_uploader('Choose an image...', type=('jpg','jpeg','png'))
  with col1:
    try:
      if source_img is None:
        st.header(':blue[Example User Ultrasound Scan]')
        uploaded_image = PIL.Image.open(os.path.join('data','raw',paths.DEFAULT_DETECTION_IMAGE))
        st.image(uploaded_image,
                 caption='Example Ultrasound Scan',
                 use_column_width=True
                )
      else: 
        st.header(':blue[User Ultrasound Scan]')
        uploaded_image = PIL.Image.open(source_img)
        st.image(source_img,
                 caption='Uploaded Ultrasound Scan',
                 use_column_width=True
                )
    except Exception as ex:
        st.error('Error occurred while opening the image.')
        st.error(ex)
  with col2:
    try:
      if source_img is None:
        get_DETECTION_col2(path=os.path.join('data','raw',paths.DEFAULT_DETECTION_GRADCAM_IMAGE), size = 224)
      else: 
        get_DETECTION_col2(path=source_img, size = 224, upload=True)

        
    except Exception as ex:
        st.error('Error occurred while opening the image.')
        st.error(ex)        

if model_type == 'Segmentation':
  col1, col2 = st.columns(2, gap='large')
  source_img = st.sidebar.file_uploader('Choose an image...', type=('jpg','jpeg','png'))
  with col1:
    try:
      if source_img is None:
        st.header(':blue[Example User Ultrasound Scan]')
        uploaded_image = PIL.Image.open('data/raw/'+paths.DEFAULT_DETECTION_IMAGE)
        st.image(uploaded_image,
                 caption='Example Ultrasound Scan',
                 use_column_width=True
                )
      else: 
        st.header(':blue[User Ultrasound Scan]')
        uploaded_image = PIL.Image.open(source_img)
        st.image(source_img,
                 caption='Uploaded Ultrasound Scan',
                 use_column_width=True
                )
    except Exception as ex:
        st.error('Error occurred while opening the image.')
        st.error(ex)
  with col2:
    try:
      if source_img is None:
        get_SEGMENTATION_col2(path='data/raw/'+paths.DEFAULT_DETECTION_GRADCAM_IMAGE, size = 256)
      else:
        get_SEGMENTATION_col2(path=source_img, size = 256, upload=True)  
        
    except Exception as ex:
        st.error('Error occurred while opening the image.')
        st.error(ex)         
