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
import shutil
import sys
import os
import time
import matplotlib.pyplot as plt
import glob
import numpy as np
import math
from skimage.transform import resize
import cv2
import pandas as pd
from PIL import Image
from pathlib import Path
import argparse
import random
import re
import torch
import matplotlib.cm as cm
from natsort import natsorted  # Use natsort for natural sorting

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def frame2tensor(frame, device):
    return torch.from_numpy(frame / 255.).float()[None, None].to(device)

def saveKpt(overlay_dir, src_kpt, dst_kpt):
    df = pd.DataFrame(columns=['src_x', 'src_y', 'dst_x', 'dst_y'])
    for ki in range(len(src_kpt)):
        df.loc[ki] = [src_kpt[ki, 0, 0], src_kpt[ki, 0, 1], dst_kpt[ki, 0, 0], dst_kpt[ki, 0, 1]]
    csv_root = os.path.join(overlay_dir, 'keypoints.csv')
    df.to_csv(csv_root, index=False)

def sg_affine(img1_file, img2_file, overlay_dir, small_image_res=1000):
    # use 255 - since the protein background is black

    img1 = cv2.imread(img1_file, 0)
    img2 = cv2.imread(img2_file, 0)

    img1_highres = img1
    img2_highres = img2

    ratio_img1 = max(img1_highres.shape[0], img1_highres.shape[1]) / float(small_image_res)
    width1 = int(img1.shape[1] / ratio_img1)
    height1 = int(img1.shape[0] / ratio_img1)
    dim1 = (width1, height1)
    # resize image
    img1 = cv2.resize(img1, dim1, interpolation=cv2.INTER_AREA)

    ratio_img2 = max(img2_highres.shape[0], img2_highres.shape[1]) / float(small_image_res)
    width2 = int(img2.shape[1] / ratio_img2)
    height2 = int(img2.shape[0] / ratio_img2)
    dim2 = (width2, height2)
    # resize image
    img2 = cv2.resize(img2, dim2, interpolation=cv2.INTER_AREA)

    # Load the SuperPoint and SuperGlue models.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running inference on device "{}"'.format(device))

    inp1 = frame2tensor(img1, device)
    inp2 = frame2tensor(img2, device)

    kpts1, kpts2 = xfeat.match_xfeat(inp1, inp2, top_k=4096)

    good_matches = []

    H, mask = cv2.findHomography(kpts1, kpts2, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
    mask = mask.flatten()

    # Get corners of the first image (image1)
    h, w = inp1.shape[:2]
    corners_inp1 = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space
    warped_corners = cv2.perspectiveTransform(corners_inp1, H)

    # Draw the warped corners in image2
    inp2_with_corners = img2.copy()
    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i - 1][0].astype(int))
        end_point = tuple(warped_corners[i][0].astype(int))
        cv2.line(inp2_with_corners, start_point, end_point, (0, 255, 0), 4)  # Using solid green for corners

    # Prepare keypoints and matches for drawMatches function
    kp1 = [cv2.KeyPoint(p[0], p[1], 5) for p in kpts1]
    kp2 = [cv2.KeyPoint(p[0], p[1], 5) for p in kpts2]
    good_matches = [cv2.DMatch(i, i, 0) for i in range(len(mask)) if mask[i]]

    good_matches_show = [good_matches]
    # Draw matches
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches_show, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    if not os.path.exists(overlay_dir):
        os.makedirs(overlay_dir)
    cv2.imwrite(os.path.join(overlay_dir, 'match_raw.jpg'), img3)

    MIN_MATCH_COUNT = 20
    if len(good_matches) >= MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M0, mask0 = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask0.ravel().tolist()

    else:
        # Use Identity Matrix When Insufficient Matches
        print(f"Insufficient matches ({len(good_matches)}). Using identity matrix as affine transformation.")
        M0 = np.array([[1, 0, 0],
                       [0, 1, 0]], dtype=np.float32)  # Identity matrix
        np.save(os.path.join(overlay_dir, 'sg_affine_init.npy'), M0)
        saveKpt(overlay_dir, np.empty((0, 1, 2)), np.empty((0, 1, 2)))
        cv2.imwrite(os.path.join(overlay_dir, 'swift_affine.jpg'), img1_highres)
        cv2.imwrite(os.path.join(overlay_dir, 'original_affine.jpg'), img2_highres)
        return 0  # Indicate that default transformation was used

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img1 = cv2.imread(img1_file)
    img2 = cv2.imread(img2_file)

    img1_highres = img1
    img2_highres = img2

    ratio_img1 = max(img1_highres.shape[0], img1_highres.shape[1]) / float(small_image_res)
    width1 = int(img1.shape[1] / ratio_img1)
    height1 = int(img1.shape[0] / ratio_img1)
    dim1 = (width1, height1)
    img1 = cv2.resize(img1, dim1, interpolation=cv2.INTER_AREA)

    ratio_img2 = max(img2_highres.shape[0], img2_highres.shape[1]) / float(small_image_res)
    width2 = int(img2.shape[1] / ratio_img2)
    height2 = int(img2.shape[0] / ratio_img2)
    dim2 = (width2, height2)
    img2 = cv2.resize(img2, dim2, interpolation=cv2.INTER_AREA)
    img4 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)
    cv2.imwrite(os.path.join(overlay_dir, 'match_inliers.jpg'), img4)

    # Select only inlier matches
    inlier_matches = [good_matches[mm] for mm in range(len(matchesMask)) if matchesMask[mm] == 1]
    good_matches = inlier_matches

    if len(good_matches) <= 10:
        M_0, _ = cv2.estimateAffine2D(src_pts, src_pts)
        np.save(os.path.join(overlay_dir, 'sg_affine_init.npy'), M_0)
        return 0

    source_matched_kpts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    target_matched_kpts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    source_matched_kpts_new = np.copy(source_matched_kpts)
    target_matched_kpts_new = np.copy(target_matched_kpts)

    x_scale = (img1_highres.shape[0] / float(img1.shape[0]))
    y_scale = (img1_highres.shape[1] / float(img1.shape[1]))
    for si in range(len(source_matched_kpts_new)):
        for sj in range(len(source_matched_kpts_new[si])):
            source_matched_kpts_new[si][sj][0] *= x_scale
            source_matched_kpts_new[si][sj][1] *= y_scale

    x_scale = (img2_highres.shape[0] / float(img2.shape[0]))
    y_scale = (img2_highres.shape[1] / float(img2.shape[1]))
    for si in range(len(target_matched_kpts_new)):
        for sj in range(len(target_matched_kpts_new[si])):
            target_matched_kpts_new[si][sj][0] *= x_scale
            target_matched_kpts_new[si][sj][1] *= y_scale

    M_inv, rigid_mask_inv = cv2.estimateAffine2D(target_matched_kpts_new, source_matched_kpts_new)
    M_0, _ = cv2.estimateAffine2D(source_matched_kpts_new, target_matched_kpts_new)
    np.save(os.path.join(overlay_dir, 'sg_affine_init.npy'), M_0)

    warped_image = cv2.warpAffine(img1_highres, M_0[:2, :3], (img2_highres.shape[1], img2_highres.shape[0]))

    cv2.imwrite(os.path.join(overlay_dir, 'swift_affine.jpg'), warped_image)
    cv2.imwrite(os.path.join(overlay_dir, 'original_affine.jpg'), img2_highres)

    saveKpt(overlay_dir, source_matched_kpts_new, target_matched_kpts_new)
    return 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run SG affine registration using XFeat on images directly in a folder."
    )
    parser.add_argument('--case_folder', type=str, default='rotated_12W_1574',
                        help='Folder containing images')
    parser.add_argument('--small_image_res', type=int, default=1000,
                        help='Maximum resolution dimension for resizing')
    args = parser.parse_args()
    
    case_folder = args.case_folder

    # Load the XFeat model once.
    xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=4096)

    # Define output directory
    image_output_dir = os.path.join(case_folder, 'affine_registration')
    if not os.path.exists(image_output_dir):
        os.makedirs(image_output_dir)

    # List all image files directly under the case folder using natsorted.
    images = natsorted([os.path.join(case_folder, f) for f in os.listdir(case_folder)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    # Process image pairs.
    for ii in range(len(images) - 1):
        fixed_root = images[ii]
        moving_root = images[ii + 1]
        print("Processing registration from fixed {} to moving {}."
              .format(os.path.basename(fixed_root), os.path.basename(moving_root)))
        # Create an overlay directory name based on the image basenames.
        overlay_dir = os.path.join(
            image_output_dir,
            '{}_to_{}'.format(os.path.splitext(os.path.basename(moving_root))[0],
                              os.path.splitext(os.path.basename(fixed_root))[0])
        )
        if not os.path.exists(overlay_dir):
            os.makedirs(overlay_dir)
        A = sg_affine(moving_root, fixed_root, overlay_dir, small_image_res=args.small_image_res)
