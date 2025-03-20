import os
import glob
import re
import cv2
import numpy as np
import pandas as pd
import argparse

# ---------------------------
# Utility Functions
# ---------------------------
def extract_number(filename):
    """
    Extract the first number from the filename.
    For example, from "image_9.jpg" returns 9.
    """
    m = re.search(r'(\d+)', filename)
    return int(m.group(1)) if m else None

def read_rotation_log(csv_path):
    """
    Reads the CSV file containing rotation data.
    Expected CSV columns: 'filename' and 'best_rotation'.
    Returns a dictionary mapping filenames (e.g. "image_9.jpg") to rotation angles in degrees.
    If the CSV is not found, prints a message and returns an empty dictionary.
    """
    rotation_dict = {}
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            rotation_dict[row['filename']] = float(row['best_rotation'])
    else:
        print(f"CSV file not found at {csv_path}. Using default rotation of 0° for all images.")
    return rotation_dict

def rotate_image(image, angle):
    """
    Rotates the image about its center by the given angle (in degrees).
    Uses OpenCV's getRotationMatrix2D and warpAffine.
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return rotated

# ---------------------------
# Main Processing
# ---------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rotate images based on rotation log CSV.")
    parser.add_argument("--input_folder", type=str, default="12W_1574",
                        help="Folder containing images and the CSV")
    parser.add_argument("--output_folder", type=str, default="rotated_12W_1574",
                        help="Folder where rotated images will be saved ")
    parser.add_argument("--csv_name", type=str, default="rotation_log.csv",
                        help="Name of the rotation CSV file in the input folder")
    parser.add_argument("--img_ext", type=str, default=".jpg",
                        help="Image file extension (default: .jpg)")
    parser.add_argument("--image_prefix", type=str, default="image_",
                        help="Image prefix (default: image_)")
    args = parser.parse_args()
    
    input_folder = args.input_folder
    output_folder = args.output_folder
    rotation_csv = os.path.join(input_folder, args.csv_name)
    img_ext = args.img_ext
    image_prefix = args.image_prefix

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read the rotation log.
    rotation_dict = read_rotation_log(rotation_csv)

    # Build a glob search pattern using the specified prefix and extension.
    search_pattern = os.path.join(input_folder, f'{image_prefix}*{img_ext}')
    image_paths = glob.glob(search_pattern)
    image_paths.sort()  # sort alphabetically

    if not image_paths:
        print("No images found in", input_folder, "with prefix", image_prefix)
        exit(1)

    for image_path in image_paths:
        filename = os.path.basename(image_path)
        # Extract the number from the filename.
        num = extract_number(filename)
        # Construct the standard filename (e.g., "image_9.jpg")
        standard_filename = f"{image_prefix}{num}{img_ext}" if num is not None else filename
        # Lookup the rotation angle; default to 0° if not found.
        angle = rotation_dict.get(standard_filename, 0.0)
        
        # Read the image.
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read {image_path}. Skipping.")
            continue

        # Rotate the image.
        rotated = rotate_image(image, angle)

        # Save the rotated image to the output folder with the same filename.
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, rotated)
        print(f"Processed {filename}: rotation {angle}° applied and saved to {output_path}")
