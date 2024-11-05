# Import libraries to be used
import os  # To work with files and directories
import numpy as np  # For numerical operations
import tensorflow as tf  # For deep learning
from tensorflow.keras.models import Sequential  # To create our model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense  # Layers for our model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img  # For image processing
import cv2  # For camera input and image processing
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Path to the folder with images
data_dir = r'C:\Users\HOME\OneDrive - Middlesex University\Documents\uno_card_AI\Uno_dataset'

# Model file name
model_path = 'Uno_model.keras'  # Save the model in the current directory

# Get class names from the dataset folders
class_names = sorted(os.listdir(data_dir))  # List of classes based on folder names

# Step 1: Prepare data
def prepare_data():
    datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        validation_split=0.2
    )
    train_data = datagen.flow_from_directory(
        data_dir,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    validation_data = datagen.flow_from_directory(
        data_dir,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    return train_data, validation_data

# Step 2: Define the model
def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(len(class_names), activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Step 3: Train the model
def train_model(model, train_data, validation_data):
    epochs = 13
    model.fit(train_data, validation_data=validation_data, epochs=epochs)
    model.save(model_path)
    messagebox.showinfo("Training", "Model trained and saved.")

# Step 4: Prediction functions
def predict_from_file(model, image_path):
    image = load_img(image_path, target_size=(64, 64))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    prediction = model.predict(image_array)
    class_index = np.argmax(prediction)
    return class_names[class_index]

def predict_from_camera(model):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture frame.")
            break
        resized_frame = cv2.resize(frame, (64, 64))
        image_array = np.expand_dims(resized_frame / 255.0, axis=0)
        prediction = model.predict(image_array)
        class_index = np.argmax(prediction)
        class_name = class_names[class_index]
        cv2.putText(frame, f'Class: {class_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# GUI setup with tkinter
def setup_gui():
    root = tk.Tk()
    root.title("UNO Card Recognition")
    root.geometry("400x400")
    root.configure(bg="pink") 

    def setup_model():
        global model
        if not os.path.exists(model_path):
            messagebox.showinfo("Info", "No model found, training a new model.")
            train_data, validation_data = prepare_data()
            model = build_model()
            train_model(model, train_data, validation_data)
        else:
            model = tf.keras.models.load_model(model_path)

    def open_file():
        file_path = filedialog.askopenfilename()
        if file_path:
            result = predict_from_file(model, file_path)
            result_label.config(text=f'Predicted UNO Card: {result}')
            image = Image.open(file_path)
            image = image.resize((150, 150), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            result_image.config(image=photo)
            result_image.image = photo

    def open_camera():
        predict_from_camera(model)

    setup_model()

    label = tk.Label(root, text="UNO Card Recognition", font=("Helvetica", 16))
    label.pack(pady=10)

    btn_file = tk.Button(root, text="Predict from File", command=open_file)
    btn_file.pack(pady=5)

    btn_camera = tk.Button(root, text="Predict from Camera", command=open_camera)
    btn_camera.pack(pady=5)

    global result_label
    result_label = tk.Label(root, text="", font=("Helvetica", 12))
    result_label.pack(pady=10)

    global result_image
    result_image = tk.Label(root)
    result_image.pack(pady=10)

    root.mainloop()

# Run the GUI application
if __name__ == '__main__':
    setup_gui()
