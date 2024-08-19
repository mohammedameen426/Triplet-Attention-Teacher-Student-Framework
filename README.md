# Triplet-Attention-Teacher-Student-Framework
Enhanced Pavement Crack Segmentation with Minimal Labeled Data: A Triplet Attention Teacher-Student Framework

Our approach effectively learns from a small set of labeled samples, minimizing the need for extensive data. The Unet architecture ensures efficient high-level feature extraction, while the triplet attention mechanism improves the refinement of model weights and enhances feature extraction, especially for indistinct crack edges. The framework combines supervised loss and uncertainty loss to train the student model, with the teacher model leveraging the exponential moving average weights of the student model. Additionally, data augmentation and input noise enhance the robustness of the framework.


Datasets used on this study:

Concrete Crack Segmentation Dataset: The concrete dataset comprises a total of 3,616 images, each sized 4032 by 33024 pixels. Among these, 1,744 images, along with their corresponding ground truth, were derived. This subset includes 458 high-resolution images of concrete surfaces captured at Middle East Technical University

Concrete dataset:https://data.mendeley.com/datasets/jwsn7tfbrp/1

DeepCrack dataset comprises 537 images with a resolution of 544 x 384 pixels, accompanied by ground truth images at the pixel level. 

DeepCrack dataset:https://github.com/yhlleo/DeepCrack

Crack500 dataset consists of 1896 training images and 1124 testing images, all at a resolution of 640 × 360 pixels. Notably, Crack500 offers a diverse range of crack shapes and widths, presenting significant challenges for crack segmentation tasks. 

Crack500 dataset: https://github.com/fyangneil/pavement-crack-detection
