# ZeroReg3D
ZeroReg: A Zero-shot Registration Pipeline for 3D Concecutive Histopathology Image Reconstruction

<img src='doc/fig1.pdf' align="center" height="500px">

## Abstract
Histological analysis plays a crucial role in understanding tissue structure and pathology. While recent advancements in registration methods have improved 2D histological analysis, they often struggle to preserve critical 3D spatial relationships, limiting their utility in both clinical and research applications. Specifically, constructing accurate 3D models from 2D slices remains challenging due to tissue deformation, sectioning artifacts, variability in imaging techniques, and inconsistent illumination. Deep learning-based registration methods have demonstrated improved performance but suffer from limited generalizability and require large-scale training data. In contrast, non-deep-learning approaches offer better generalizability but often compromise on accuracy. In this paper, we introduce ZeroReg3D, a zero-shot registration pipeline that integrates zero-shot deep learning-based keypoint matching and non-deep-learning alignment techniques to effectively mitigate deformation and sectioning artifacts without requiring extensive training data. Comprehensive evaluations demonstrate that our pairwise 2D image registration method improves alignment accuracy by approximately 10% over baseline methods, outperforming existing strategies in both accuracy and robustness. High-fidelity 3D reconstructions further validate the effectiveness of our approach, establishing ZeroReg3D as a reliable framework for precise 3D reconstruction from consecutive 2D histological images.

## Installation
~~~
conda create -n ZeroReg python=3.8
conda activate ZeroReg
pip install -r requirements.txt
~~~

## Use
[optional] rotation preprocessing
~~~
python rotation_preprocess.py --case_folder <Path/To/Image Folder/> --angle_step <rotation degree>
python rotate_images.py --input_folder <your case folder> --output_folder <Path/to/Output Folder>
~~~

To get the initial affine registration images
~~~
python Step1_affine_registration.py --case_folder <Path/To/Image Folder/>
python Step2_get_3D_inital.py -case_folder <Path/To/Image Folder/>
~~~

To get the final nonrigid registration images
~~~
python Step3_nonrigid_registration.py --case_folder <Path/To/Image Folder/>
python Step4_get_3D_final.py -case_folder <Path/To/Image Folder/>
~~~






