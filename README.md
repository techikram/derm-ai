
# Application of Convolutional Neural Networks for Melanoma Classification

**Author:** Ikram Ahdadouche El Idrissi
**Date:** September 2024

## Project Overview
Melanoma is a dangerous form of skin cancer with a high global mortality rate. Despite technological advancements, early-stage diagnosis still requires further improvement. This project aims to contribute a user-friendly tool for dermatologists, utilizing a deep Convolutional Neural Network (CNN) model to efficiently and swiftly identify melanoma. The tool is accessible via a web-based interface hosted at [https://derm-ai.udg.edu/derm-ai](https://derm-ai.udg.edu/derm-ai).

## Project Structure
This project was divided into two main phases:
1. **Model Training and Testing**: Focused on training a CNN model based on EfficientNet-B3 architecture, chosen for its pattern-capturing ability without over-specialization.
2. **Web-based API Development**: Built a Django API from scratch, incorporating both backend logic and frontend interface.

## Data Source and Preprocessing
- **Datasets**: Used the ISIC Challenge datasets (ISIC 2019 and ISIC 2020), available on [Kaggle ISIC 2020](https://www.kaggle.com/datasets/cdeotte/jpeg-melanoma-192x192) and [Kaggle ISIC 2019](https://www.kaggle.com/datasets/agsam23/isic-2019-challenge).
- **Preprocessing**: Steps included data cleansing, reduction, transformation, enrichment, and validation. Artifacts like band-aids and ink marks were removed, and data augmentation techniques were applied to reduce overfitting risks.

## Model Configuration
The following hyper-parameters were tuned and selected for the CNN model:

| Hyper-parameter      | Value         |
|----------------------|---------------|
| Initial Learning rate| 0.00001       |
| Batch size           | 128           |
| Activation function  | Swish         |
| Number of epochs     | 35            |
| Weight decay         | 0.00001       |
| Number of workers    | 8             |
| Learning rate decay  | Cosine decay  |

## Training and Inference
Training and testing were conducted with a classifier hosted on a UdG server. Results were visualized in the Wandb framework. Additionally, an independent script was developed for inference on the image classifier, analyzing input images and outputting results in a CSV file with probabilities for classifications (benign, suspicious, malignant, or melanoma).

## Web Interface
- **Django API**: The Django API interacts bidirectionally with the classification script. Both are hosted in the same environment for efficiency.
- **User Experience**: The website features a home page for login/registration. Users can upload images and view historical results on their profile page. For privacy, images are temporarily stored and deleted after analysis.

## Results
The model demonstrated an ROC-AUC of 0.83, with a sensitivity of 96% at a 95% fixed threshold, proving effective in distinguishing between melanoma and non-melanoma cases.

## Conclusion
This project aims to optimize the melanoma diagnostic process for physicians and patients. The web platform enables fast and efficient melanoma diagnosis, significantly simplifying usage for dermatologists and patients alike.

