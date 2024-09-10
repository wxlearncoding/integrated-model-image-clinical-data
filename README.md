# integrated-model-image-clinical-data
This is the supplementary code for the paper "A Comparison of an Integrated and Image-Only Deep Learning Model for Predicting the Disappearance of Indeterminate Pulmonary Nodules".

## Input data
### 1 image preprocessing 
- The lung window setting was performed based on [WW: 1600 HU, WL: -700 HU].
- CT scans were interpolated to a voxel size of 1×1×1 mm using trilinear interpolation.
- Each lung nodule was saved separately in the center of a 3D cube of 32×32×32 mm<sup>3</sup>.
- Image format is npy

### 2 Clinical data preprocessing
Clinical data: participant demographics  
- numerical variables (age and pack-years) -> normalized them to the range of [0,1]
- categorical variables (gender and smoking status) -> one-hot encoding


## Model training and testing
If you want to test your data, please run the code "train_test_main.py".




