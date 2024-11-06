# UNO-Recognition

# ================

This repository contains code and resources for building an UNO card recognition model. The project includes image resizing, data preprocessing, model training, and a GUI application for recognizing UNO cards via file upload or live camera feed.

## Repository Structure
* `resizeImg.py`: Resizes images in the dataset to a target width, maintaining the original aspect ratio.
* `dataset_preprocessor.py`: Applies augmentations and transformations to the images to create a richer dataset for training.
* `train.py`: Defines and trains a convolutional neural network (CNN) model for UNO card classification.
* `app.py`: Provides a GUI to predict UNO card classes using a pre-trained model, with options to select an image file or use a live camera feed.
## Getting Started
### Prerequisites
* Python 3.8 or later
* Install required libraries with:
> bash
```
pip install tensorflow pillow opencv-python numpy tkinter
```
### Dataset
UNO card images are stored in the `dataset` folder. Each class has its own subfolder, e.g., `dataset/0_red`, `dataset/1_yellow`, etc. The dataset folder will be used for resized images, and `UNO_dataset` will store augmented data.

## Scripts Overview
1. Image Resizing - `resizeImg.py`
This script resizes all images in the un_dataset directory to a target width of 540 pixels, preserving the aspect ratio. Resized images are saved in the dataset directory.

#### Usage:

> bash code
```
python resizeImg.py
```
2. Data Preprocessing - `dataset_preprocessor.py`
The script applies various transformations and augmentations to each image to enhance dataset variety, creating rotated, zoomed, and filtered versions of each image.

#### Usage:

> bash code
```
python dataset_preprocessor.py
```
3. Model Training - `train.py`
This script defines, compiles, and trains a CNN model on the augmented images in the `UNO_dataset` directory. The model architecture includes several convolutional, pooling, and dropout layers to improve accuracy.

#### Usage:

> bash code
```
python train.py
```
The trained model will be saved as Model.keras in the project directory.

4. GUI Prediction App - `app.py`
A Tkinter GUI application allows you to predict UNO card classes by uploading an image file or using a live camera feed. The GUI displays the predicted card name on the screen.

#### Usage:

> bash code
```
python app.py
```
* **Predict from File**: Allows you to select an image file from your computer and display the prediction.
* **Predict from Camera**: Uses the live camera feed to recognize UNO cards in real-time.