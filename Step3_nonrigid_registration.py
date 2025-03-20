from typing import Union
import pathlib
import os
import argparse

### External Imports ###
import numpy as np
import torch as tc
import matplotlib.pyplot as plt
import natsort
### DeeperHistReg Imports ###
import deeperhistreg
import glob

from deeperhistreg.dhr_input_output.dhr_loaders import pil_loader
from deeperhistreg.dhr_input_output.dhr_loaders import tiff_loader
from deeperhistreg.dhr_pipeline.registration_params import default_initial_nonrigid


### Run Registration ###
def run(source_path, target_path, output_path):
    source_path = pathlib.Path(source_path)
    target_path = pathlib.Path(target_path)
    output_path = pathlib.Path(output_path)

    ### Define Params ###
    registration_params: dict = default_initial_nonrigid()
    
    registration_params['loading_params']['loader'] = 'pil'  # For .jpg or .png formats
    registration_params['run_initial_registration'] = False
    registration_params['initial_registration_params']['save_results'] = False
    save_displacement_field: bool = True  # Whether to save the displacement field
    copy_target: bool = True  # Whether to copy the target
    delete_temporary_results: bool = False  # Whether to keep the temporary results
    case_name: str = f"{source_path.stem}_{target_path.stem}"  # Used only if the temporary_path is important
    temporary_path: Union[str, pathlib.Path] = output_path / f"{source_path.stem}_{target_path.stem}_TEMP"

    ### Create Config ###
    config = dict()
    config['source_path'] = source_path
    config['target_path'] = target_path
    config['output_path'] = output_path
    config['registration_parameters'] = registration_params
    config['case_name'] = case_name
    config['save_displacement_field'] = save_displacement_field
    config['copy_target'] = copy_target
    config['delete_temporary_results'] = delete_temporary_results
    config['temporary_path'] = temporary_path

    ### Run Registration ###
    deeperhistreg.run_registration(**config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run DeeperHistReg registration on a sequence of images."
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="rotated_12W_1574",
        help="Base folder containing the image directories (default: 'rotated_12W_1574')"
    )
    parser.add_argument(
        "--source_subfolder",
        type=str,
        default="affine_registration",
        help="Subfolder with source images (default: 'registration_3D_raw_xfeat_image')"
    )
    parser.add_argument(
        "--output_subfolder",
        type=str,
        default="nonrigid_displacement",
        help="Subfolder to store registration outputs (default: 'registration_DeeperHistReg')"
    )

    args = parser.parse_args()

    folder = args.folder
    source_folder = pathlib.Path(os.path.join(folder, args.source_subfolder))
    output_path = pathlib.Path(os.path.join(folder, args.output_subfolder))
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Get sorted list of image paths
    image_paths = natsort.natsorted(source_folder.glob("image_*.jpg"))
    if not image_paths:
        raise ValueError(f"No images found in {source_folder} with pattern 'image_*.jpg'.")

    # Calculate middle index so that for images 1..N, we pick the middle image
    middle_index = (len(image_paths) - 1) // 2
    print(f"Number of images: {len(image_paths)} | Middle index: {middle_index} | Middle image: {image_paths[middle_index].name}")

    for i in range(len(image_paths)):
        if i < middle_index:
            # Register in sequential order before the middle index (e.g., image_1 to image_2, image_2 to image_3)
            source_path = image_paths[i]
            target_path = image_paths[i + 1]
        elif i > middle_index:
            # Register in reverse order after the middle index (e.g., image_4 to image_3, image_5 to image_4)
            source_path = image_paths[i]
            target_path = image_paths[i - 1]
        else:
            # Optionally, you could skip registering the middle image against itself
            continue

        run(source_path, target_path, output_path)
