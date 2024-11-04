import os
from PIL import Image, ImageFilter, ImageEnhance

# Define the root directory and the output directory
ROOT_DIR = r'dataset'
OUTPUT_DIR = r'UNO_dataset'

# Image formats of interest
IMAGE_FORMATS = ('.jpg', '.png', '.jpeg')

# Define the filters
def define_filters():
    return {
        # "Blur": ImageFilter.BLUR,
        # "Contour": ImageFilter.CONTOUR,
        # "Detail": ImageFilter.DETAIL,
        # "Edge Enhance": ImageFilter.EDGE_ENHANCE,
        # "Emboss": ImageFilter.EMBOSS,
        # "Find Edges": ImageFilter.FIND_EDGES,
        # "Sharpen": ImageFilter.SHARPEN,
        "Brightness Increase": lambda img: ImageEnhance.Brightness(img).enhance(1.5),
        "Contrast Increase": lambda img: ImageEnhance.Contrast(img).enhance(1.5),
        "Identity": lambda img: img
    }

# Define the additional transformations
def define_transformations(img):
    return [
        ("original", img),
        ("grayscale", img.convert("L")),
        ("blur", img.filter(ImageFilter.BLUR)),
        # ("contour", img.filter(ImageFilter.CONTOUR)),
        ("edge_enhance", img.filter(ImageFilter.EDGE_ENHANCE)),
        ("sharpen", img.filter(ImageFilter.SHARPEN)),
        ("detail", img.filter(ImageFilter.DETAIL)),
        ("contrast", ImageEnhance.Contrast(img).enhance(2.0)),
        ("brightness", ImageEnhance.Brightness(img).enhance(1.5)),
        ("rotate", img.rotate(45)),
        ("flip_horizontal", img.transpose(Image.FLIP_LEFT_RIGHT)),
        ("flip_vertical", img.transpose(Image.FLIP_TOP_BOTTOM))
    ]

# Process the images with filters and transformations
def process_images(root_dir, output_dir):
    filters = define_filters()
    
    # Traverse directories
    for subdir, dirs, files in os.walk(root_dir):
        relative_path = os.path.relpath(subdir, root_dir)
        output_path = os.path.join(output_dir, relative_path)
        os.makedirs(output_path, exist_ok=True)  # Create output path if it doesn't exist
        
        for file in files:
            if file.endswith(IMAGE_FORMATS):
                file_path = os.path.join(subdir, file)
                try:
                    img = Image.open(file_path).convert("RGB")
                    
                    # Apply predefined filters
                    for filter_name, filter_func in filters.items():
                        try:
                            if isinstance(filter_func, ImageFilter.Filter):
                                filtered_img = img.filter(filter_func)
                            elif callable(filter_func):
                                filtered_img = filter_func(img)
                            else:
                                raise TypeError(f"Unknown filter type for {filter_name}")
                            
                            # Save the filtered image
                            filename, file_extension = os.path.splitext(file)
                            new_filename = f"{filename}_{filter_name}{file_extension}"
                            new_file_path = os.path.join(output_path, new_filename)
                            filtered_img.save(new_file_path)
                            print(f"Saved: {new_file_path}")
                        except Exception as e:
                            print(f"Error applying filter {filter_name} on {file_path}: {e}")

                    # Apply additional transformations
                    for transformation_name, transformed_img in define_transformations(img):
                        try:
                            # Save each transformed image
                            filename, file_extension = os.path.splitext(file)
                            new_filename = f"{filename}_{transformation_name}{file_extension}"
                            new_file_path = os.path.join(output_path, new_filename)
                            transformed_img.save(new_file_path)
                            print(f"Saved: {new_file_path}")
                        except Exception as e:
                            print(f"Error applying transformation {transformation_name} on {file_path}: {e}")
                            
                except Exception as e:
                    print(f"Error opening or processing {file_path}: {e}")

if __name__ == "__main__":
    process_images(ROOT_DIR, OUTPUT_DIR)
