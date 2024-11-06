# UNO-Recognition


This repository contains code and resources for building an UNO card recognition model. The project includes image resizing, data preprocessing, model training, and a GUI application for recognizing UNO cards via file upload or live camera feed.

## Repository Structure
* `dataset`: A folder, dataset, that contains the UNO cards images.
* `resizeImg.py`: Resizes images in the dataset to a target width, maintaining the original aspect ratio.
* `dataset_preprocessor.py`: Applies augmentations and transformations to the images to create a richer dataset for training.
* `train.py`: Defines and trains a convolutional neural network (CNN) model for UNO card classification.
* `app.py`: Provides a GUI to predict UNO card classes using a pre-trained model, with options to select an image file or use a live camera feed.
* `Model.keras` & `Model(2).keras`: trained models, products of `train.py`, to be used by `app.py`.
## Getting Started
### Prerequisites
* Python 3.8 or later
* Install required libraries with:
> bash code
```
pip install tensorflow pillow opencv-python numpy tkinter
```
### Dataset
UNO card images are stored in the `dataset` folder. Each class has its own subfolder, e.g., `dataset/Red_0`, `dataset/Yellow_Draw_2`, etc. The dataset folder will be utilized by `resizeImg.py` and `dataset_preprocessor.py` to create `UNO_dataset` which will store the augmented data.

## Scripts Overview
1. Image Resizing - `resizeImg.py`
- This script resizes all images in the `dataset` directory to a target width of 540 pixels, preserving the aspect ratio. The aim is to reduce the dataset large size reducing the training workload.

#### Usage:

> bash code
```
python resizeImg.py
```
2. Data Preprocessing - `dataset_preprocessor.py`
- The script applies various transformations and augmentations to each image to enhance dataset variety, creating rotated, zoomed, and filtered versions of each image.

#### Usage:

> bash code
```
python dataset_preprocessor.py
```
3. Model Training - `train.py`
- This script defines, compiles, and trains a CNN model on the augmented images in the `UNO_dataset` directory. The model architecture includes several convolutional, pooling, and dropout layers to improve accuracy. The trained model will be saved as `Model.keras` in the project directory.

#### Usage:

> bash code
```
python train.py
```

4. GUI Prediction App - `app.py`
- A Tkinter GUI application allows you to predict UNO card classes by uploading an image file or using a live camera feed. The GUI displays the predicted card name on the screen.

#### Usage:

> bash code
```
python app.py
```
* **Predict from File**: Allows you to select an image file from your computer and display the prediction.
* **Predict from Camera**: Uses the live camera feed to recognize UNO cards in real-time.

## Credits

This project was developed by a team of three contributors:

* <a href="https://github.com/saki3110">Sakina Taygaully<a>
* <a href="https://github.com/gt663">Hans Nursing<a>
* <a href="https://github.com/EdwinAbdonShayo">Edwin Shayo<a>