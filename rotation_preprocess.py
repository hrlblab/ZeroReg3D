import os
import torch 
import glob
import numpy as np
import cv2
import pandas as pd
import argparse
from PIL import Image
from natsort import natsorted

def frame2tensor(frame, device):
    # Normalize image intensity values to [0,1] and convert to tensor (1,1,H,W)
    return torch.from_numpy(frame / 255.).float()[None, None].to(device)

def rotate_image(image, angle):
    """
    Rotate image by the given angle (in degrees).
    Returns the rotated image and the rotation matrix (2x3).
    """
    if angle % 360 == 0:
        return image.copy(), np.eye(2, 3, dtype=np.float32)
    
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    R = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, R, (w, h), flags=cv2.INTER_LINEAR)
    return rotated, R

def count_inliers(img1_file, img2_file, xfeat, rotation_angle=0, fixed_rotation=0, small_image_res=1000):
    """
    Count the number of inliers after keypoint matching between two images.
    """
    # Read images in color.
    img1_color = cv2.imread(img1_file, cv2.IMREAD_COLOR)
    img2_color = cv2.imread(img2_file, cv2.IMREAD_COLOR)

    if img1_color is None:
        print(f"Error reading {img1_file}")
        return 0
    if img2_color is None:
        print(f"Error reading {img2_file}")
        return 0

    # Convert images to grayscale.
    img1_original_gray = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2_original_gray = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    # Rotate the moving image as specified.
    img1_rotated_gray, _ = rotate_image(img1_original_gray, rotation_angle)

    # Rotate the fixed image if needed.
    if fixed_rotation % 360 != 0:
        img2_rotated_gray, _ = rotate_image(img2_original_gray, fixed_rotation)
    else:
        img2_rotated_gray = img2_original_gray

    # Resize images if they exceed the small_image_res.
    ratio_img1 = max(img1_rotated_gray.shape[0], img1_rotated_gray.shape[1]) / float(small_image_res)
    ratio_img1 = max(ratio_img1, 1.0)
    width1 = int(img1_rotated_gray.shape[1] / ratio_img1)
    height1 = int(img1_rotated_gray.shape[0] / ratio_img1)
    img1_resized_gray = cv2.resize(img1_rotated_gray, (width1, height1), interpolation=cv2.INTER_AREA)

    ratio_img2 = max(img2_rotated_gray.shape[0], img2_rotated_gray.shape[1]) / float(small_image_res)
    ratio_img2 = max(ratio_img2, 1.0)
    width2 = int(img2_rotated_gray.shape[1] / ratio_img2)
    height2 = int(img2_rotated_gray.shape[0] / ratio_img2)
    img2_resized_gray = cv2.resize(img2_rotated_gray, (width2, height2), interpolation=cv2.INTER_AREA)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running inference on device "{device}" with moving rotation {rotation_angle}° and fixed rotation {fixed_rotation}°.')

    # Convert images to tensors.
    inp1 = frame2tensor(img1_resized_gray, device)
    inp2 = frame2tensor(img2_resized_gray, device)

    # Extract keypoints and perform matching using XFeat.
    kpts1, kpts2 = xfeat.match_xfeat(inp1, inp2, top_k=4096)
    if len(kpts1) == 0 or len(kpts2) == 0:
        print("No keypoints detected.")
        return 0

    # Compute Homography using RANSAC and count inliers.
    H, mask = cv2.findHomography(kpts1, kpts2, cv2.USAC_MAGSAC, 3.5, maxIters=1000, confidence=0.999)
    if mask is not None:
        mask = mask.flatten()
        num_inliers = int(np.sum(mask))
    else:
        num_inliers = 0

    return num_inliers

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process images for rotation alignment using XFeat.")
    parser.add_argument('--case_folder', type=str, default='12W_1574',
                        help='Folder containing images (default: output)')
    parser.add_argument('--small_image_res', type=int, default=1000,
                        help='Maximum resolution dimension for resizing (default: 1000)')
    parser.add_argument('--angle_step', type=int, default=90,
                        help='Rotation angle step in degrees (default: 1)')
    args = parser.parse_args()
    
    case_folder = args.case_folder

    # Load the XFeat model once.
    xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=4096)
    print(f"Processing images directly under {case_folder}")

    # List all image files (jpg, jpeg, png) directly under the case folder using natsorted.
    images = natsorted(
        [os.path.join(case_folder, f) for f in os.listdir(case_folder)
         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    )

    rotation_log = []

    # Process image pairs.
    for ii, moving_root in enumerate(images):
        filename = os.path.basename(moving_root)

        if ii == 0:
            print(f"Processing first image {filename}. No rotation needed.")
            rotation_log.append({
                'filename': filename,
                'best_rotation': 0,
                'best_inliers': None
            })
            continue

        fixed_root = images[ii - 1]
        fixed_filename = os.path.basename(fixed_root)
        
        # Get the best rotation of the fixed image from the log.
        previous_best_rotation = rotation_log[-1]['best_rotation']
        print(f'\nProcessing pair: {filename} (moving) vs {fixed_filename} (fixed rotated by {previous_best_rotation}°).')

        best_inliers = -1
        best_rotation = 0

        # Evaluate candidate rotations for the moving image.
        for angle in range(0, 360, args.angle_step):
            num_inliers = count_inliers(
                moving_root,
                fixed_root,
                xfeat,
                rotation_angle=angle,
                fixed_rotation=previous_best_rotation,
                small_image_res=args.small_image_res
            )
            if num_inliers > best_inliers:
                best_inliers = num_inliers
                best_rotation = angle

        print(f'Best rotation for {filename} relative to {fixed_filename} (fixed rotated by {previous_best_rotation}°): {best_rotation}° with {best_inliers} inliers.')
        rotation_log.append({
            'filename': filename,
            'best_rotation': best_rotation,
            'best_inliers': best_inliers
        })

    # Save the rotation log as CSV.
    if rotation_log:
        rotation_log_df = pd.DataFrame(rotation_log)
        csv_path = os.path.join(case_folder, 'rotation_log.csv')
        rotation_log_df.to_csv(csv_path, index=False)
        print(f"\nSaved rotation log CSV at {csv_path}\n")
    else:
        print("No rotation data collected.\n")
