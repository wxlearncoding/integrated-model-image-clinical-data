# integrated-model-image-clinical-data
This is the supplementary code for the paper "**A Comparison of an Integrated and Image-Only Deep Learning Model for Predicting the Disappearance of Indeterminate Pulmonary Nodules**".

## Input data
Please follow and cite our paper for data preprocessing: Wang J, Sourlos N, Zheng S, et al. Preparing CT imaging datasets for deep learning in lung nodule analysis: Insights from four well-known datasets. Heliyon. 2023;9(6):e17104. Published 2023 Jun 16. doi:10.1016/j.heliyon.2023.e17104 [https://pubmed.ncbi.nlm.nih.gov/37484314/].

### 1 image preprocessing 
- The lung window setting was performed based on WW: 1600 HU and WL: -700 HU
- CT scans were interpolated to a voxel size of 1×1×1 mm using trilinear interpolation
- Each lung nodule was saved separately in the center of a 3D cube of 32×32×32 mm<sup>3</sup>
- Image format is npy

### 2 Clinical data preprocessing
Clinical data: participant demographics  
- numerical variables (age and pack-years) -> normalized them to the range of [0,1]
- categorical variables (gender and smoking status) -> one-hot encoding

## Model training and testing
For testing your data, please run the code "train_test_main.py".

## Feature visualization
For feature importance, please run the code "vis_model_integrated.py".




