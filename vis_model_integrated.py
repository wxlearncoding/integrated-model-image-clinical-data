'''
@Author: Wei Tang, Jingxuan Wang

@Contact: j.wang02@umcg.nl
'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from monai.transforms import Compose, LoadImaged, AddChanneld, ScaleIntensityd, ToTensord, RandFlipd, \
    RandGaussianNoised, RandAffined, EnsureChannelFirstd, Resized, RandAdjustContrastd, RandRotated, RandFlipd,CenterSpatialCropd
from monai.data import Dataset, DataLoader
from monai.metrics import get_confusion_matrix
import monai
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import time
import datetime
from torch.nn.functional import interpolate
import matplotlib.pyplot as plt
import numpy as np
import itertools
import glob
import shap
import matplotlib
matplotlib.use('Agg')  # or 'TkAgg', 'GTK3Agg', etc.

# visualization using SHAP

class ResNet18_merge(nn.Module):

    def __init__(self):
        super(ResNet18_merge, self).__init__()

        # Initialize the ResNet18 model
        self.resnet18 = monai.networks.nets.resnet18(spatial_dims=3, n_input_channels=1, num_classes=2)

        # Remove the last fully connected layer
        self.features = nn.Sequential(*list(self.resnet18.children())[:-1])

        self.mlp1 = nn.Linear(512,128)
        self.mlp2 = nn.Linear(128,16)
        self.mlp3 = nn.Linear(23,2)

    def forward(self, x, embeddings):

        batch_size = 4 # hardcoded

        # Forward pass through the modified network
        x = self.features(x)
        # Flatten the output
        x = torch.flatten(x, 1)
        x = self.mlp1(x)
        x = self.mlp2(x)

        # Inputting clinical data and also handling empty embeddings
        if len(embeddings) == 0:
            embeddings = torch.zeros([batch_size,7]) 

        combined = torch.cat([x, embeddings.float()], dim=1)

        x = self.mlp3(combined)

        return x


def clean_embeddings(embeddings):
    for i, emb in enumerate(embeddings):
        try:
            embeddings[i] = emb[0]
        except:
            embeddings[i] = np.zeros_like(
                embeddings[next(index for index, item in enumerate(embeddings) if item is not None)])
    return embeddings


device = torch.device("cpu")

model = ResNet18_merge().to(device)
model_name = 'ensemble_3ss'

df = pd.read_csv('../5fold_cv_4.csv')
count = len(df)

keys = ["image"]

pd_num_cat = pd.read_csv('../processed_features_840.csv') # please use your file location

all_shap_values_0, all_shap_values_1, all_shap_values, all_combined_inputs = [], [], [], []

for i in range(1, 6):

    def model_mlp3_input(x, embeddings):
        x = model.features(x)
        x = torch.flatten(x, 1)
        x = model.mlp1(x)
        x = model.mlp2(x)
        if len(embeddings) == 0:
            embeddings = torch.zeros([x.size(0), 7])  # Adjust for batch size
        combined_input = torch.cat([x, embeddings.float()], dim=1)
        return combined_input

    print("............START........fold{}.................".format(i))

    validation_files = df.loc[df['fold'] == i, 'filename'].tolist()
    validation_labels = df.loc[df['fold'] == i, 'label'].tolist()

    val_ids = [int(i.split('_')[-2]) for i in validation_files]

    val_embeddings = clean_embeddings(
        [pd_num_cat[pd_num_cat['Nodule_ID'] == i][['Age', 'PackYears', 'Gender_1', 'Gender_2',
                                                   'SmokingStatus_1', 'SmokingStatus_2', 'SmokingStatus_3']].values for i in val_ids])

    validation_data = [{"image": img, "label": label, "embedding": emb} for img, label, emb in
                       zip(validation_files, validation_labels, val_embeddings)]

    validation_transforms = Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Resized(keys=keys, spatial_size=(32, 32, 32), mode="trilinear"),
        ToTensord(keys=keys),
    ])

    validation_dataset = Dataset(data=validation_data, transform=validation_transforms)
    validation_loader = DataLoader(validation_dataset, batch_size=20, shuffle=False)
    dataloaders = {'val': validation_loader}

    '''
    Model loading
    '''
    model_save_path = r'...'  # please use your file location
    model_save = glob.glob(os.path.join(model_save_path, "best_model_fold{}_epoch*.pth".format(i)))[0] # find the model for each fold
    state_dict = torch.load(model_save, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()  # Set model to evaluate mode

    '''
    Visualization
    '''
    # Initialize lists to store SHAP values and combined inputs
    batch_combined_inputs = []
    batch_shap_values = []
    batch_shap_values_0 = []
    batch_shap_values_1 = []

    for batch_data in dataloaders['val']:

        image_tensor, labels, clinical_data_tensor = batch_data['image'].to(device), batch_data['label'].to(device), batch_data['embedding'].to(device)

        # nodule_address = image_tensor.meta['filename_or_obj'][0]
        # print(nodule_address)
        # nodule_label = nodule_address.split("\\")[3]
        # nodule_id = nodule_address.split("\\")[-1].split("_")[2]

        combined_input = model_mlp3_input(image_tensor, clinical_data_tensor)

        # Initialize the SHAP explainer for the MLP layer
        # explainer = shap.GradientExplainer(model.mlp3, combined_input)
        explainer = shap.DeepExplainer(model.mlp3, combined_input)
        # explainer = shap.KernelExplainer(model.mlp3, combined_input)

        # Compute SHAP values for the current batch
        shap_values = explainer.shap_values(combined_input)

        # Append SHAP values and inputs for this batch!!!  e.g., batch_size = 20, each fold has 168 nodules
        batch_combined_inputs.append(combined_input.detach().cpu().numpy())
        batch_shap_values.append(shap_values) # for two classes
        batch_shap_values_0.append(shap_values[0])  # Assuming shap_values[0] is for the positive class # shap_values[1] for negative class
        batch_shap_values_1.append(shap_values[1])


    # Convert lists to numpy arrays
    fold_shap_values_0 = np.concatenate(batch_shap_values_0, axis=0)
    fold_shap_values_1 = np.concatenate(batch_shap_values_1, axis=0)
    fold_shap_values = [fold_shap_values_0,fold_shap_values_1]
    fold_combined_inputs = np.concatenate(batch_combined_inputs, axis=0)

    all_shap_values_0.append(fold_shap_values_0)
    all_shap_values_1.append(fold_shap_values_1)
    all_combined_inputs.append(fold_combined_inputs)
    all_shap_values.append(fold_shap_values)

    # Create custom feature names
    num_image_features = 16
    num_clinic_features = 7
    clinic_names = ['Age', 'PackYears', 'Gender_1', 'Gender_2', 'SmokingStatus_1', 'SmokingStatus_2', 'SmokingStatus_3']
    feature_names = [f'Image_feature_{i + 1}' for i in range(num_image_features)] + clinic_names

    # Generate bar plot for both two classes
    plt.figure()
    shap.summary_plot(fold_shap_values,fold_combined_inputs,feature_names=feature_names,plot_type="bar",class_names=['Non-resolving','Resolving'],show=False,max_display=25)
    plt.savefig(os.path.join(model_save_path,f'shap_summary_bar_fold{i}.png'))

    # check wrong display
    # shap.summary_plot(fold_shap_values_0,all_combined_inputs,feature_names=feature_names,plot_type="bar",class_names=['Non-resolving','Resolving'],show=False,max_display=10)
    # plt.savefig(f'../result/shap_summary_bar_0_fold{i}.png')
    # shap.summary_plot(fold_shap_values_1,all_combined_inputs,feature_names=feature_names,plot_type="bar",class_names=['Non-resolving','Resolving'],show=False,max_display=10)
    # plt.savefig(f'../result/shap_summary_bar_1_fold{i}.png')

    print(f"summary bar plot for fold{i}")
    print("............END........fold{}.................".format(i))
    print()

plt.figure()
shap.summary_plot(np.concatenate(np.array(all_shap_values_0),axis=0), np.concatenate(np.array(all_combined_inputs),axis=0),feature_names=feature_names, show=False, max_display=25)
plt.savefig(os.path.join(model_save_path,'shap_summary_beeswarm_all_0.png'))
plt.figure()
shap.summary_plot(np.concatenate(np.array(all_shap_values_1),axis=0), np.concatenate(np.array(all_combined_inputs),axis=0),feature_names=feature_names, show=False, max_display=25)
plt.savefig(os.path.join(model_save_path,'shap_summary_beeswarm_all_1.png'))

plt.figure()
shap.summary_plot([np.concatenate(np.array(all_shap_values_0),axis=0),np.concatenate(np.array(all_shap_values_1),axis=0)], np.concatenate(np.array(all_combined_inputs),axis=0),feature_names=feature_names,plot_type="bar",class_names=['Non-resolving','Resolving'],show=False,max_display=25)
plt.savefig(os.path.join(model_save_path,'shap_summary_bar_all.png'))
