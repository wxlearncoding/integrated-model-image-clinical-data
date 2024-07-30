# integrated-model-image-clinical-data
This is the supplementary code for the paper "Integrated Deep Learning Model for Prediction of Disappearance of Indeterminate Pulmonary Nodules in Low-dose Chest CT Scans".

## Input data
### 1 image preprocessing 
The lung window setting was performed based on [WW: 1600 HU, WL: -700 HU]. Since CT scans in the NELSON and ImaLife data derive from different years, medical centers, and CT scanners, we performed image preprocessing on both NELSON datasets to reduce the variability in slice thicknesses and image resolution. CT scans were interpolated to a voxel size of 1×1×1 mm using trilinear interpolation [22]. To reduce noise caused by the background surrounding the nodules, we extracted the nodules from the whole CT scans using the centroid coordinates labelled by radiologists. Each lung nodule was saved separately in the center of a 3D cube of 32×32×32 mm3.

### 2 Clinical data preprocessing
Clinical data: participant demographics  
numerical variables (age and pack-years) -> normalized them to the range of [0,1]
categorical variables (gender and smoking status) -> one-hot encoding


## Model training and testing
If you want to test your data, please run the code "train_test_main.py".




