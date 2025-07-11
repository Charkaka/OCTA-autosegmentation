import cv2
import numpy as np
import os
from glob import glob

# Input and output directories
input_dir = "./clientData"
output_dir = "./clientDataFlattened"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get all image file paths from the input directory
image_paths = glob(os.path.join(input_dir, "*.*"))  # Adjust the pattern if needed (e.g., "*.png" or "*.jpg")

# Process each image
for image_path in image_paths:
    # Read the image
    color_image = cv2.imread(image_path)

    # Check if the image was successfully loaded
    if color_image is None:
        print(f"Failed to load image: {image_path}")
        continue

    # Convert to grayscale
    grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # Get the output file path
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)

    # Save the grayscale image
    cv2.imwrite(output_path, grayscale_image)

    print(f"Processed and saved: {output_path}")

print("All images have been processed.")