# BreastTumourClassificationAndSegmentationWithGradCAM

Breast cancer remains a significant global health challenge, with approximately 2.3 million new cases diagnosed each year, according to the World Health Organization (WHO). The impact is staggering, as breast cancer is the most common cancer among women worldwide, contributing to nearly 1 in 4 cancer cases in females. Tragically, it is responsible for over 685,000 deaths annually. In the United States alone, an estimated 1 in 8 women will develop invasive breast cancer during their lifetime. These statistics underscore the urgent need for effective treatments and, ultimately, a cure. Advancements in research and medical science are crucial in addressing this public health concern, improving survival rates, and enhancing the quality of life for millions of individuals affected by breast cancer.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Workflow](#workflow)
- [Results](#results)
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

* bc.ipynb contains the end-to-end code for Image classification
* bc.ipynb contains the end-to-end code for Semantic Segmentation.
* app.py contains the streamlit app code.
* utils and helper folders contain modules for app.py

## Results
For the first task, we tried Transfer learning using popular CNN models like VGG19, ResNet, EfficientNet as well vision model as Vision Transformer.
| model           |   Precision_Normal |   Precision_Benign |   Precision_Malignant |   Recall_Normal |   Recall_Benign |   Recall_Malignant |   Recall_BM |
|:----------------|-------------------:|-------------------:|----------------------:|----------------:|----------------:|-------------------:|------------:|
| EfficientNetB7  |               0.72 |               0.86 |                  0.82 |            0.85 |            0.83 |               0.78 |        0.81 |
| ResNet152V2     |               0.71 |               0.89 |                  0.86 |            0.89 |            0.88 |               0.73 |        0.8  |
| VGG19           |               0.74 |               0.86 |                  0.62 |            0.93 |            0.73 |               0.73 |        0.73 |
| EfficientNetV2S |               0.53 |               0.85 |                  0.69 |            0.89 |            0.69 |               0.66 |        0.68 |
| ConvNeXtBase    |               0.7  |               0.84 |                  0.7  |            0.78 |            0.8  |               0.73 |        0.76 |
| vit_b16         |               0.27 |               0.75 |                  0.45 |            0.81 |            0.07 |               0.73 |        0.4  |
| vit_b16         |               0.31 |               0.73 |                  0.62 |            0.81 |            0.4  |               0.56 |        0.48 |
| vit_b16         |               0.47 |               0.76 |                  0.55 |            0.52 |            0.59 |               0.78 |        0.69 |
| vit_b16         |               0.38 |               0.83 |                  0.52 |            0.74 |            0.39 |               0.78 |        0.58 |

![alt text](https://github.com/krunalgedia/BreastTumourClassificationAndSegmentationWithGradCAM/blob/main/images_app/benign.gif)


## References
[1]: Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief. 2020 Feb;28:104863. DOI: 10.1016/j.dib.2019.104863


