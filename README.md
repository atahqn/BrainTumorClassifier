# CS484 Introduction to Computer Vision Term Project:
#           Brain Tumor Classifier 

This project aims to classify brain tumors using Convolutional Neural Networks (CNNs) and pre-trained models in PyTorch. The code is in progress and in the final version, we plan to add some augmented images to the dataset to ensure the classes are evenly distributed. Also, the optimizer will be changed to Adam optimizer for better performance.

## Data

The dataset is taken from Kaggle. You can download it from this [link](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri).

## Libraries Used

The following libraries are used in this project:

- os
- time
- random
- numpy
- PIL
- matplotlib
- seaborn
- torch
- torchvision
- sklearn
- tqdm

## Setting Up CUDA

If a GPU is available, the code is configured to use it for training the models, else it falls back on the CPU.

## Image Transformation

Images are resized to 150x150 pixels, and also a random horizontal flip is applied. Then they are converted to tensors and normalized.

## Dataset Preparation

Train and test datasets are loaded from the specified paths. Both datasets are combined and then split into the train, validation, and test sets. The split is 80% for training, 10% for validation, and 10% for testing.

## Models

The following models are used:

- Custom CNN
- Pre-trained ResNet18
- Non pre-trained ResNet18
- Pre-trained EfficientNet B0
- Non pre-trained EfficientNet B0

All layers except the last one are frozen for pre-trained models, and a new last layer is added for the 4-class classification task.

## Training, Inference, and Testing

The code includes functions for training the model, making predictions on the validation set, and evaluating the model on the test set. The loss function used is CrossEntropyLoss.
In progress, SGD with momentum optimizer is used. In the final version, the optimizer changed to the Adam optimizer.

## Results

Detailed results and visualizations can be found in the project report.

