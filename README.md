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
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview

[Provide a brief overview of your project, including its purpose, goals, and any relevant context.]
The goal of this project is to provide a web application through which one could upload the MRI scan of a patient's breast to know if the breast has tumor cells, and if yes, which one among benign and malignant and show the confidence of the ML model in predicting the classes.
Another goal is to perform instance segmentation on the MRI scan to know the location of the tumour cells if present.
Both these predictions must be accompanied by GradCAM of the final (attention) layer to know where the focus of the model was while predicting the tumor cells.

## Installation

[Include instructions for installing any necessary dependencies or setting up the environment. You may want to provide code snippets or refer to a separate document for detailed installation steps.]

```bash
# Example installation command
pip install -r requirements.txt

