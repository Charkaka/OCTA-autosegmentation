import os
from PIL import Image

def check_image_sizes(folder):
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            path = os.path.join(folder, fname)
            try:
                with Image.open(path) as img:
                    print(f"{fname}: {img.size} (width x height)")
            except Exception as e:
                print(f"{fname}: Error - {e}")

if __name__ == "__main__":
    folder = r"datasets\background_images"
    check_image_sizes(folder)