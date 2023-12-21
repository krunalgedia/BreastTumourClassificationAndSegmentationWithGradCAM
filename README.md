# BreastTumourClassificationAndSegmentationWithGradCAM

Breast cancer remains a significant global health challenge, with approximately 2.3 million new cases diagnosed each year, according to the World Health Organization (WHO). The impact is staggering, as breast cancer is the most common cancer among women worldwide, contributing to nearly 1 in 4 cancer cases in females. Tragically, it is responsible for over 685,000 deaths annually. In the United States alone, an estimated 1 in 8 women will develop invasive breast cancer during their lifetime. These statistics underscore the urgent need for effective treatments and, ultimately, a cure. Advancements in research and medical science are crucial in addressing this public health concern, improving survival rates, and enhancing the quality of life for millions of individuals affected by breast cancer.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Workflow](#workflow)
- [Results](#results)
- [More ideas](#More ideas)
- [Dependencies](#dependencies)
- [License](#license)
- [Contact](#contact)
- [References](#references)

## Project Overview

[Provide a brief overview of your project, including its purpose, goals, and any relevant context.]
The goal of this project is to develop a real-time web application to
* Perform classification of the uploaded Ultrasound scan of a patient's breast to know if the breast has benign or malignant tumor cells along with confidence in prediction. 
* Perform semantic segmentation of the Ultrasound scan of a patient's breast to know the location of the tumor cells if present.
* Provide additional insights for both classification and segmentation to know what the model focuses on while performing the prediction.

## Installation

[Include instructions for installing any necessary dependencies or setting up the environment. You may want to provide code snippets or refer to a separate document for detailed installation steps.]

```bash
# Example installation command
pip install -r requirements.txt

# Run Web Application
streamlit run app.py
```

## Data
The data collected at baseline include breast ultrasound images among women in ages between 25 and 75 years old. This data was collected in 2018. The number of patients is 600 female patients. The dataset consists of 780 images with an average image size of 500*500 pixels. The images are in PNG format. The ground truth images are presented with original images. The images are categorized into three classes, which are normal, benign, and malignant. [1]

## Workflow
1. Importing data
2. Observing data
3. Loading data in appropriate form.
4. Training different models and comparing metrics.
5. Running predictions on the model
6. Running Grad CAM insights on the final attention layer

* notebooks/classification_breast_cancer.ipynb contains the end-to-end code for Image classification
* notebooks/segmentation_breast_cancer.ipynb contains the end-to-end code for Semantic Segmentation.
* app.py contains the streamlit app code.
* utils and helper folders contain modules for app.py

## Results
For the first task of image classification, we tried Transfer learning using popular CNN models like VGG19, ResNet, and EfficientNet as well as Vision models as Vision Transformer. Using average recall of Benign and Malignant tumors, we found EfficientNet to be the best-performing one on the test set. The models were implemented in tf-keras with callbacks implemented for overtraining.
The following table gives the gist of the performance.

| model           |   Precision_Normal |   Precision_Benign |   Precision_Malignant |   Recall_Normal |   Recall_Benign |   Recall_Malignant |   Recall_BM |
|:----------------|-------------------:|-------------------:|----------------------:|----------------:|----------------:|-------------------:|------------:|
| EfficientNetB7  |               0.72 |               0.86 |                  0.82 |            0.85 |            0.83 |               0.78 |        0.81 |
| ResNet152V2     |               0.71 |               0.89 |                  0.86 |            0.89 |            0.88 |               0.73 |        0.8  |
| VGG19           |               0.74 |               0.86 |                  0.62 |            0.93 |            0.73 |               0.73 |        0.73 |
| EfficientNetV2S |               0.53 |               0.85 |                  0.69 |            0.89 |            0.69 |               0.66 |        0.68 |
| ConvNeXtBase    |               0.7  |               0.84 |                  0.7  |            0.78 |            0.8  |               0.73 |        0.76 |
| vit_b16         |               0.27 |               0.75 |                  0.45 |            0.81 |            0.07 |               0.73 |        0.4  |

The web app gives the prediction using the EfficientNet model. Following is the demo for each of the three classes:
![Image 1](https://github.com/krunalgedia/BreastTumourClassificationAndSegmentationWithGradCAM/blob/main/images_app/opening_page.png) | ![Image 2](https://github.com/krunalgedia/BreastTumourClassificationAndSegmentationWithGradCAM/blob/main/images_app/malignant_detect.png) | ![Image 3](https://github.com/krunalgedia/BreastTumourClassificationAndSegmentationWithGradCAM/blob/main/images_app/normal_detect.png)
--- | --- | ---
Benign Scan | Malignant Scan | Normal Scan


For the second task of image semantic segmentation, we fine-tuned the pre-trained models used in the medical literature like UNet, ResUNet, ResUNet with Attention. Using dice score and Intersection over Union (IoU) as a metric, all the models were found to be performing similarly. We finally decided to go with ResUNet with Attention since it provided negligible gains in metrics over the other two models. The models were implemented in tf-keras.

The web app gives the segmentation prediction using the EfficientNet model. Following is the demo for each of the three classes:
![Image 1](https://github.com/krunalgedia/BreastTumourClassificationAndSegmentationWithGradCAM/blob/main/images_app/benign.gif) | ![Image 2](https://github.com/krunalgedia/BreastTumourClassificationAndSegmentationWithGradCAM/blob/main/images_app/malignant.gif) | ![Image 3](https://github.com/krunalgedia/BreastTumourClassificationAndSegmentationWithGradCAM/blob/main/images_app/normal.gif)
--- | --- | ---
Benign Scan | Malignant Scan | Normal Scan

For the third task, we used Grad-CAM which uses Class Activation Map along with gradient information to give insights into the model. The final heatmap is overlaid on the input image as shown in the web app for both classification and segmentation predictions.

<img src="https://github.com/krunalgedia/BreastTumourClassificationAndSegmentationWithGradCAM/blob/main/images_app/difficult.gif" alt="Image" width="400"/> Such Grad-CAM heatmaps can help greatly in giving insights to the user about the model's focus in decision-making. This can help in cases like the one shown on the left side where the segmentation prediction is not the best. However, looking at the GradCAM heatmap shows the areas the model focussed on and we see the focus included the region of the tumor but the model. Thus, even though the model fails here, such insights can help the doctor to detect the probable areas of the tumor.

## More ideas

For the classification model, maybe cross combination or concatenating transformer vision model with CNN models could work even better since transformer models are known to capture the global picture while CNN models look more at the local picture. 

For the segmentation model, maybe stacking of models could be tried.

## Dependencies

This project uses the following dependencies:

- **Python:** 3.10.12
- **Tensorflow:** 2.15.0
- **Streamlit:** 1.28.2 

- [Trained Classification model](https://www.dropbox.com/scl/fi/lnp23cdyo4eq0nckn2vtq/Detection_model?rlkey=ll5wy8fhw4mb9fopk83xdkbn0&dl=0)
- [Trained Segmentation model](https://www.dropbox.com/scl/fi/cq1tescfxr1uvp8z8fb2g/Segmentation_model?rlkey=s81dvvzzdlj0q92kjxvno78vr&dl=0)
  
## Contact

Feel free to reach out if you have any questions, suggestions, or feedback related to this project. I'd love to hear from you!

- **LinkedIn:** [Krunal Gedia](https://www.linkedin.com/in/krunal-gedia-00188899/)

## References
[1]: Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief. 2020 Feb;28:104863. DOI: 10.1016/j.dib.2019.104863


