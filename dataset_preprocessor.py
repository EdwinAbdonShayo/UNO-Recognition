import os
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

ROOT_DIR = r'dataset'
OUTPUT_DIR = r'UNO_dataset'
IMAGE_FORMATS = ('.jpg', '.png', '.jpeg')

# Define transformations and filters
def apply_transformations_and_filters(img):
    return [
        ("original", img),
        ("rotate_5", img.rotate(5)),
        ("rotate_neg5", img.rotate(-5)),
        ("small_zoom_in", img.resize((int(img.width * 1.05), int(img.height * 1.05))).crop((int(img.width * 0.025), int(img.height * 0.025), int(img.width * 1.025), int(img.height * 1.025)))),
        ("small_zoom_out", img.resize((int(img.width * 0.95), int(img.height * 0.95))).crop((0, 0, img.width, img.height))),
        ("hue_shift", ImageEnhance.Color(img).enhance(1.05)),  
        ("light_noise", img.filter(ImageFilter.GaussianBlur(0.5))), 
        ("translate_up", ImageOps.expand(img, (0, 10, 0, 0), fill="black").crop((0, 0, img.width, img.height))),
        ("translate_left", ImageOps.expand(img, (10, 0, 0, 0), fill="black").crop((0, 0, img.width, img.height))),
    ]

def ensure_orientation(image):
    width, height = image.size
    if height < width:
        return image.rotate(90, expand=True)
    return image

# Process and save images with transformations and filters
def process_images(root_dir, output_dir):
    for subdir, dirs, files in os.walk(root_dir):
        relative_path = os.path.relpath(subdir, root_dir)
        output_path = os.path.join(output_dir, relative_path)
        os.makedirs(output_path, exist_ok=True)
        
        for file in files:
            if file.endswith(IMAGE_FORMATS):
                file_path = os.path.join(subdir, file)
                try:
                    img = Image.open(file_path).convert("RGB")
                    
                    for name, transformed_img in apply_transformations_and_filters(img):
                        try:
                            # Ensure orientation with shorter side as the width
                            transformed_img = ensure_orientation(transformed_img)
                            
                            # Save each transformed image
                            filename, file_extension = os.path.splitext(file)
                            new_filename = f"{filename}_{name}{file_extension}"
                            new_file_path = os.path.join(output_path, new_filename)
                            transformed_img.save(new_file_path)
                            print(f"Saved: {new_file_path}")
                        except Exception as e:
                            print(f"Error saving {name} on {file_path}: {e}")
                            
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    process_images(ROOT_DIR, OUTPUT_DIR)
