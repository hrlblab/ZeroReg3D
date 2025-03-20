#!/usr/bin/env python
# Copyright 2018 University of Basel, Center for medical Image Analysis and Navigation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import glob
import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import warp
import cv2 as cv2
from pathlib import Path

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order.
    http://nedbatchelder.com/blog/200712/human_sorting.html
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Perform 3D registration reconstruction using SG affine transformation matrices and warping."
    )
    parser.add_argument(
        "--case_folder", type=str, default="rotated_12W_1574",
        help="Folder containing case subfolders (default: rotated_12W_1574)"
    )
    parser.add_argument(
        "--image_pattern", type=str, default="image_*.jpg",
        help="Pattern for image filenames in each case folder (default: image_*.jpg)"
    )
    args = parser.parse_args()

    case_folder = args.case_folder
    image_pattern = args.image_pattern

    # Define input and output directories.
    image_input_dir = case_folder
    # The affine matrices are expected to be saved in subfolders under this directory.
    image_output_dir = os.path.join(case_folder, 'affine_registration')
    reconstruction_dir = os.path.join(case_folder, 'affine_3D_images')

    if not os.path.exists(reconstruction_dir):
        os.makedirs(reconstruction_dir)

    # Get images matching the pattern.
    images = glob.glob(os.path.join(image_input_dir, image_pattern))
    images.sort(key=natural_keys)

    if not images:
        print("No images found in", image_input_dir, "with pattern", image_pattern)
        sys.exit(1)

    # Choose the middle image as reference.
    middle_idx = int(len(images) / 2)
    print("Middle index:", middle_idx)
    middle_image = plt.imread(images[middle_idx])[:, :, :3]

    # Process each image based on its position relative to the middle image.
    for ii in range(len(images)):
        now_idx = ii

        if now_idx < middle_idx:
            # Compute the cumulative transformation from the current image up to the reference.
            M_new = np.zeros((3, 3))
            M_new[2, 2] = 1.0

            for ri in range(now_idx, middle_idx):
                # Use the full base filename to construct the folder name.
                fixed_name = os.path.splitext(os.path.basename(images[ri]))[0]
                moving_name = os.path.splitext(os.path.basename(images[ri+1]))[0]
                affine_root = os.path.join(
                    image_output_dir,
                    f'{moving_name}_to_{fixed_name}',
                    'sg_affine_init.npy'
                )
                M1 = np.zeros((3, 3))
                M1[:2, :3] = np.load(affine_root)
                M1[2, 2] = 1.0

                if ri == now_idx:
                    M_new = M1.copy()
                else:
                    M_new = M_new.dot(M1)

            now_image = plt.imread(images[now_idx])[:, :, :3]
            img1_affine = warp(now_image, M_new, output_shape=(middle_image.shape[0], middle_image.shape[1]))
            new_root = images[now_idx].replace(image_input_dir, reconstruction_dir)
            plt.imsave(new_root, img1_affine)
            print(f"Saved warped image for index {now_idx} to {new_root}")

        elif now_idx > middle_idx:
            # Compute the cumulative transformation from the reference up to the current image.
            M_new = np.zeros((3, 3))
            M_new[2, 2] = 1.0

            for ri in range(middle_idx, now_idx):
                fixed_name = os.path.splitext(os.path.basename(images[ri]))[0]
                moving_name = os.path.splitext(os.path.basename(images[ri+1]))[0]
                affine_root = os.path.join(
                    image_output_dir,
                    f'{moving_name}_to_{fixed_name}',
                    'sg_affine_init.npy'
                )
                M1 = np.zeros((3, 3))
                M1[:2, :3] = np.load(affine_root)
                M1[2, 2] = 1.0

                if ri == middle_idx:
                    M_new = M1.copy()
                else:
                    M_new = M_new.dot(M1)

            # Invert the cumulative transformation to map the image back to the reference frame.
            affine_matrix_inv = cv2.invertAffineTransform(M_new[:2, :])
            M_new[:2, :] = affine_matrix_inv

            now_image = plt.imread(images[now_idx])[:, :, :3]
            img1_affine = warp(now_image, M_new, output_shape=(middle_image.shape[0], middle_image.shape[1]))
            new_root = images[now_idx].replace(image_input_dir, reconstruction_dir)
            plt.imsave(new_root, img1_affine)
            print(f"Saved warped image for index {now_idx} to {new_root}")

        else:
            # For the middle (reference) image, simply copy it over.
            new_root = images[middle_idx].replace(image_input_dir, reconstruction_dir)
            plt.imsave(new_root, middle_image)
            print(f"Saved middle image to {new_root}")
