# ZeroReg3D
ZeroReg: A Zero-shot Registration Pipeline for 3D Concecutive Histopathology Image Reconstruction

## Abstract
Histological analysis plays a crucial role in understanding tissue structure and pathology. While recent advancements in registration methods have improved 2D histological analysis, they often struggle to preserve critical 3D spatial relationships, limiting their utility in both clinical and research applications. Specifically, constructing accurate 3D models from 2D slices remains challenging due to tissue deformation, sectioning artifacts, variability in imaging techniques, and inconsistent illumination. Deep learning-based registration methods have demonstrated improved performance but suffer from limited generalizability and require large-scale training data. In contrast, non-deep-learning approaches offer better generalizability but often compromise on accuracy. In this paper, we introduce ZeroReg3D, a zero-shot registration pipeline that integrates zero-shot deep learning-based keypoint matching and non-deep-learning alignment techniques to effectively mitigate deformation and sectioning artifacts without requiring extensive training data. Comprehensive evaluations demonstrate that our pairwise 2D image registration method improves alignment accuracy by approximately 10% over baseline methods, outperforming existing strategies in both accuracy and robustness. High-fidelity 3D reconstructions further validate the effectiveness of our approach, establishing ZeroReg3D as a reliable framework for precise 3D reconstruction from consecutive 2D histological images.

# Installation
~~~
conda create -n ZeroReg python=3.8
~~~
