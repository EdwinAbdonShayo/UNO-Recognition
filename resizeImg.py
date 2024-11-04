import os
from PIL import Image

def resize_images_in_folders(main_folder_path, output_folder, target_width=540):
    # Loop through each subfolder in the main folder
    for root, dirs, files in os.walk(main_folder_path):
        # Determine relative path to maintain directory structure
        relative_path = os.path.relpath(root, main_folder_path)
        output_path = os.path.join(output_folder, relative_path)
        os.makedirs(output_path, exist_ok=True)  # Create output path if it doesn't exist

        for file_name in files:
            # Check if the file is an image by its extension
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Construct the full file path
                file_path = os.path.join(root, file_name)
                try:
                    # Open the image
                    with Image.open(file_path) as img:
                        # Calculate the new height based on the original aspect ratio
                        aspect_ratio = img.height / img.width
                        target_height = int(target_width * aspect_ratio)
                        
                        print(f"Height = {target_height}")
                        # Resize the image to the new size
                        resized_img = img.resize((target_width, target_height), Image.LANCZOS)
                        
                        # Save the resized image to the output directory, preserving file extension
                        output_file_path = os.path.join(output_path, file_name)
                        resized_img.save(output_file_path)
                    print(f"Resized and saved: {output_file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

# Example usage
main_folder = './un_dataset'
output_folder = './dataset'
resize_images_in_folders(main_folder, output_folder)
