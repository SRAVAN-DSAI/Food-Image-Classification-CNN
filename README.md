# Food-Image-Classification-CNN

This repository contains a deep learning model built using TensorFlow and Keras for classifying food images into various categories. The model uses Convolutional Neural Networks (CNNs) to classify images into 10 selected food categories.

## Project Overview

This project involves the following key steps:

1. **Dataset Collection**: A dataset of food images is loaded from a local directory.
2. **Data Preprocessing**: Images are resized, normalized, and split into training and test sets.
3. **Model Building**: A CNN model is created using TensorFlow/Keras.
4. **Training**: The model is trained using the training dataset with data augmentation to improve generalization.
5. **Evaluation**: Model performance is evaluated using accuracy, loss, confusion matrix, and classification reports.

## Dataset

The dataset used in this project contains images of food items. The images are stored in subdirectories, each representing a different category. The dataset is located in the `food_data` directory and should contain subfolders representing each food category.

### Categories

The script uses the first 10 food categories from the dataset for training the model. You can change this by modifying the `selected_categories` list in the code.

## Requirements

- Python
- TensorFlow >= 2.0
- Keras
- NumPy
- Matplotlib
- scikit-learn
- Seaborn

You can install the necessary dependencies using pip:
```
pip install tensorflow numpy matplotlib scikit-learn seaborn
```
## Usage
Clone the repository:
```
git clone https://github.com/SRAVAN-DSAI/Food-Image-Classification-CNN
cd food-image-classification
```
Prepare your dataset:

Download or organize a food image dataset.
Place the dataset inside the food_data directory. Each category should be in its own subfolder within food_data.
Run the script:
```
python food_image_classification.py
```
The script will:

Load and preprocess the images.
Train the CNN model.
Display training and validation accuracy/loss plots.
Evaluate the model on the test data and print the classification report.
Show a confusion matrix of predictions.

##Model Architecture
The CNN model consists of the following layers:

3 convolutional layers with ReLU activation.
2 max-pooling layers to downsample the feature maps.
1 fully connected layer with 64 neurons and ReLU activation.
A dropout layer to reduce overfitting.
An output layer with softmax activation for multi-class classification.
## Results
The model outputs the following:

Accuracy and loss plots for both training and validation sets.
Test accuracy after evaluating the model on the test dataset.
A classification report that includes precision, recall, and F1-score for each class.
A confusion matrix to visualize the model's performance across all categories.
