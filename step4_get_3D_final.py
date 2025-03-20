import SimpleITK as sitk
import numpy as np
from PIL import Image
import os
import re
import shutil
import argparse

# -------------------------
# Helper Functions
# -------------------------

def prepare_displacement_field(displacement_field):
    """Prepare a displacement field for transformation."""
    displacement_field_np = sitk.GetArrayFromImage(displacement_field)
    if displacement_field.GetDimension() == 3 and displacement_field_np.shape[0] == 2:
        displacement_field_np = np.transpose(displacement_field_np, (1, 2, 0))
    displacement_field_np = displacement_field_np.astype(np.float64)
    displacement_field_vector = sitk.GetImageFromArray(displacement_field_np, isVector=True)
    displacement_field_vector = sitk.Cast(displacement_field_vector, sitk.sitkVectorFloat64)
    return displacement_field_vector

def process_images(base_displacement_field_path, base_image_path, output_path,
                   available_images, middle_image_number, target_image,
                   input_prefix, output_prefix, input_extension, output_extension):
    """
    Process a target image by building a transformation chain from the target image
    to the middle image based on available images.
    """
    # Determine chain from target_image to middle_image_number
    if target_image < middle_image_number:
        chain = [img for img in available_images if target_image <= img <= middle_image_number]
    else:
        chain = [img for img in available_images if middle_image_number <= img <= target_image]
        chain = sorted(chain, reverse=True)

    if len(chain) < 2:
        print(f"Not enough intermediate images to build transformation chain for image {target_image}")
        return

    print(f"Building transformation chain for image {target_image}: {chain}")

    transforms = []
    # Build displacement field paths for each consecutive pair in the chain.
    for idx in range(len(chain) - 1):
        source = chain[idx]
        target = chain[idx + 1]
        # Expected folder structure can be adapted as needed.
        # Here we assume the displacement field folder names use the input_prefix and the image numbers.
        displacement_field_subpath = (
            f'{input_prefix}{source}_{input_prefix}{target}_TEMP/'
            f'{input_prefix}{source}_{input_prefix}{target}/Results_Final/displacement_field.mha'
        )
        displacement_field_path = os.path.join(base_displacement_field_path, displacement_field_subpath)
        if not os.path.exists(displacement_field_path):
            print(f"Displacement field not found for pair ({source}, {target}): {displacement_field_path}")
            continue
        try:
            displacement_field = sitk.ReadImage(displacement_field_path)
            displacement_field_prepared = prepare_displacement_field(displacement_field)
            transform = sitk.DisplacementFieldTransform(displacement_field_prepared)
            transforms.append(transform)
        except Exception as e:
            print(f"Error processing displacement field for pair ({source}, {target}) from {displacement_field_path}: {e}")
            continue

    if not transforms:
        print(f"No valid transforms found for image {target_image}")
        return

    # For images after the middle, reverse the order of transforms.
    if target_image > middle_image_number:
        transforms = list(reversed(transforms))

    composite_transform = sitk.CompositeTransform(transforms)

    # Build the input image filename.
    input_filename = f'{input_prefix}{target_image}{input_extension}'
    image_path = os.path.join(base_image_path, input_filename)
    if not os.path.exists(image_path):
        print(f"Image not found, skipping: {image_path}")
        return

    try:
        # Open and convert the image.
        image = Image.open(image_path)
        image_np = np.array(image)
        if image_np.ndim == 3 and image_np.shape[2] == 3:
            image_sitk = sitk.GetImageFromArray(image_np, isVector=True)
        else:
            image_sitk = sitk.GetImageFromArray(image_np)

        # Resample the image using the composite transform.
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(image_sitk)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(composite_transform)

        deformed_image_sitk = resampler.Execute(image_sitk)
        deformed_image_np = sitk.GetArrayFromImage(deformed_image_sitk)
        deformed_image_np = np.clip(deformed_image_np, 0, 255).astype(np.uint8)
        deformed_image = Image.fromarray(deformed_image_np)

        # Build the output filename.
        output_filename = f'{output_prefix}{target_image}{output_extension}'
        output_file = os.path.join(output_path, output_filename)
        deformed_image.save(output_file)
        print(f"Processed image {target_image} to align with middle image {middle_image_number}")
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

def find_available_images(base_image_path, input_prefix, input_extension, pattern=None):
    """
    Scans the base image directory and returns a sorted list of available image numbers.
    The regex pattern is built from the input_prefix and input_extension if not provided.
    """
    image_numbers = []
    if pattern is None:
        pattern = rf'{input_prefix}(\d+){input_extension}'
    regex = re.compile(pattern)
    for filename in os.listdir(base_image_path):
        match = regex.match(filename)
        if match:
            try:
                number = int(match.group(1))
                image_numbers.append(number)
            except ValueError:
                continue
    if not image_numbers:
        raise ValueError("No images found that match the given pattern.")
    image_numbers.sort()
    return image_numbers

# -------------------------
# Main Script
# -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process images using displacement fields to build a transformation chain for registration."
    )
    parser.add_argument('--input_prefix', type=str, default="image_",
                        help="Prefix for input images (default: 'image_')")
    parser.add_argument('--output_prefix', type=str, default="image_",
                        help="Prefix for output images (default: 'image_')")
    parser.add_argument('--input_extension', type=str, default=".jpg",
                        help="Extension for input images (default: '.jpg')")
    parser.add_argument('--output_extension', type=str, default=".jpg",
                        help="Extension for output images (default: '.jpg')")
    parser.add_argument('--base_displacement_field_path', type=str,
                        default="rotated_12W_1574/registration_DeeperHistReg",
                        help="Base directory for displacement fields (default: rotated_12W_1574/registration_DeeperHistReg)")
    parser.add_argument('--base_image_path', type=str,
                        default="rotated_12W_1574/registration_3D_raw_xfeat_image",
                        help="Directory containing input images (default: rotated_12W_1574/registration_3D_raw_xfeat_image)")
    parser.add_argument('--output_path', type=str,
                        default="rotated_12W_1574/nonrigid_3D_image",
                        help="Output directory for processed images (default: rotated_12W_1574/nonrigid_3D_image)")
    parser.add_argument('--pattern', type=str, default=None,
                        help="Regex pattern to find available images. If not provided, defaults to a pattern built from input_prefix and input_extension.")
    args = parser.parse_args()

    # Assign parameters from args.
    input_prefix = args.input_prefix
    output_prefix = args.output_prefix
    input_extension = args.input_extension
    output_extension = args.output_extension
    base_displacement_field_path = args.base_displacement_field_path
    base_image_path = args.base_image_path
    output_path = args.output_path

    # Create the output directory if it doesn't exist.
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    try:
        available_images = find_available_images(base_image_path, input_prefix, input_extension, args.pattern)
        print(f"Available images: {available_images}")
    except Exception as e:
        print(f"Error finding available images: {e}")
        exit(1)

    # Determine the middle image using the sorted available images.
    middle_index = len(available_images) // 2
    middle_image_number = available_images[middle_index]
    print(f"Middle image determined as: {middle_image_number}")

    # Process images before the middle image.
    images_before = [img for img in available_images if img < middle_image_number]
    for img_num in images_before:
        process_images(base_displacement_field_path, base_image_path, output_path,
                       available_images, middle_image_number, target_image=img_num,
                       input_prefix=input_prefix, output_prefix=output_prefix,
                       input_extension=input_extension, output_extension=output_extension)

    # Process images after the middle image.
    images_after = [img for img in available_images if img > middle_image_number]
    for img_num in images_after:
        process_images(base_displacement_field_path, base_image_path, output_path,
                       available_images, middle_image_number, target_image=img_num,
                       input_prefix=input_prefix, output_prefix=output_prefix,
                       input_extension=input_extension, output_extension=output_extension)

    # Copy the middle image directly.
    middle_filename = f"{input_prefix}{middle_image_number}{input_extension}"
    middle_image_path = os.path.join(base_image_path, middle_filename)
    output_middle_path = os.path.join(output_path, middle_filename)
    if os.path.exists(middle_image_path):
        shutil.copyfile(middle_image_path, output_middle_path)
        print(f"Middle image {middle_filename} copied directly to the output folder.")
    else:
        print(f"Middle image not found: {middle_image_path}")
