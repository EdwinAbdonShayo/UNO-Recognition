# Import libraries to be used
import os  # To work with files and directories
import numpy as np  # For numerical operations
import tensorflow as tf  # For deep learning
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img  # For image processing
import cv2  # For camera input and image processing
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Path to the folder with images
data_dir = r'dataset'

# Model file name
model_path = 'Model.keras'  # Save the model in the current directory

# Get class names from the dataset folders
class_names = sorted(os.listdir(data_dir))  # List of classes based on folder names

# Step 4: Prediction functions
def predict_from_file(model, image_path):
    image = load_img(image_path, target_size=(255, 340))
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
        resized_frame = cv2.resize(frame, (255, 340))
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
    root.geometry("400x600")
    root.configure(bg="#404040") 

    def setup_model():
        global model
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
        else:
            messagebox.showerror("Error", "Model not found. Please train the model first.")

    def open_file():
        file_path = filedialog.askopenfilename()
        if file_path:
            result = predict_from_file(model, file_path)
            result_label.config(text=f'Predicted UNO Card: {result}')
            image = Image.open(file_path)
            image = image.resize((255, 340), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            result_image.config(image=photo)
            result_image.image = photo

    def open_camera():
        predict_from_camera(model)

    setup_model()

    label = tk.Label(root, text="UNO Card Recognition", font=("Comfortaa", 16, "bold"), bg="#404040", fg="#ffffff")
    label.pack(pady=10)

    btn_file = tk.Button(root, text="Predict from File", command=open_file, font=("Comfortaa", 12))
    btn_file.pack(pady=5)

    btn_camera = tk.Button(root, text="Predict from Camera", command=open_camera, font=("Comfortaa", 12))
    btn_camera.pack(pady=5)

    global result_label
    result_label = tk.Label(root, text="", font=("Comfortaa", 12), bg="#404040", fg="#ffffff")
    result_label.pack(pady=10)

    global result_image
    result_image = tk.Label(root, bg="#404040")
    result_image.pack(pady=10)

    root.mainloop()

# Run the GUI application
if __name__ == '__main__':
    setup_gui()
