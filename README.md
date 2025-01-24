# Image Classifier for Flower Species

## Overview
This project aims to develop a sophisticated image classifier using a Convolutional Neural Network (CNN) built with PyTorch. The classifier is trained to identify different species of flowers from a dataset containing 102 categories. This initiative forms part of a larger educational program focused on deep learning techniques and image classification.

## Technologies Used
- Python
- Convolutional Neural Network (CNN)
- PyTorch

## Aim
The project seeks to:
1. Load and preprocess an image dataset of flowers.
2. Train an image classifier on the dataset.
3. Utilize the trained classifier to make predictions on new images.
4. Develop a command-line application to recognize and classify flower species.

## Architectures Briefing
### Convolutional Neural Networks (CNN)
- **Convolutional Layers**: Extract features from the input image using filters.
- **ReLU Activation Functions**: Introduce non-linearity to the network.
- **Max Pooling Layers**: Reduce spatial dimensions and retain important features.
- **Fully Connected Layers**: Perform classification based on the extracted features.
- **Dropout**: Regularize the network to prevent overfitting.

## Model Architecture
- **Conv2D**: Two-dimensional convolutional layers to extract image features.
- **ReLU**: Activation functions for non-linear transformations.
- **MaxPooling2D**: Layers to reduce spatial dimensions.
- **Fully Connected**: Layers to perform final classifications.
- **Dropout**: Layers to prevent overfitting.

## Results
After training the model, its performance was evaluated on the test dataset. Here are the key performance metrics:
- **Accuracy**: 90.4% 

## How It Works
1. **Input Images**: Users submit images of flowers.
2. **Preprocessing**: Images are preprocessed and normalized for the model.
3. **Classification**: The trained CNN model classifies the flower species.
4. **Output**: The model outputs the predicted species of the flower.

## Usage
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/flower-image-classifier.git
   cd flower-image-classifier

