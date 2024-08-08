"""
@Author: Wei Tang, Jingxuan Wang

@Contact: j.wang02@umcg.nl
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from monai.transforms import Compose, LoadImaged, AddChanneld, ScaleIntensityd, ToTensord, RandFlipd, RandGaussianNoised,RandAffined,EnsureChannelFirstd,Resized, RandAdjustContrastd,RandRotated,RandFlipd,CenterSpatialCropd
from monai.data import Dataset,DataLoader
from monai.metrics import get_confusion_matrix
import monai
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.calibration import calibration_curve
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
import itertools

import matplotlib
matplotlib.use('Agg')


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

        # Handling missing data in embeddings
        if embeddings.size(0) == 0:
            embeddings = torch.zeros([batch_size, 7])
        else:
            # Handling numerical data (e.g., Age, Pack-Years)
            for i, column in enumerate([age_index, pack_years_index]):  # specify indices based on your dataframe
                median_value = torch.nanmedian(embeddings[:, column])
                embeddings[:, column] = torch.where(torch.isnan(embeddings[:, column]),
                                                    median_value,
                                                    embeddings[:, column])
                
            # Impute missing categorical data (e.g., Gender, Smoking Status) with mode
            for category_indices in [gender_indices, smoking_indices]:  # specify one-hot indices
                mode_value = torch.mode(embeddings[:, category_indices], dim=0)[0]
                missing_mask = torch.isnan(embeddings[:, category_indices]).all(dim=1)
                embeddings[missing_mask, category_indices] = mode_value
        
        combined = torch.cat([x, embeddings.float()], dim=1)

        x = self.mlp3(combined)

        return x


def train_val_model(device, running_date):

    Excel = '5fold_cv.csv'

    df = pd.read_csv(Excel)
    count = len(df)

    keys = ["image"]

    pd_num_cat = pd.read_csv('clinical_features_840.csv')

    best_metric_container = {
        'best_true_label': [],
        'best_predicted_label': [],
        'best_metric': [],
        'best_probability': []
    }
    
    results_df = pd.DataFrame(columns=['Fold', 'Epoch', 'Phase', 'Loss', 'Accuracy'])
    
    def clean_embeddings(embeddings):
        for i, emb in enumerate(embeddings):
            try:
                embeddings[i] = emb[0]
            except:
                embeddings[i] = np.zeros_like(embeddings[next(index for index, item in enumerate(embeddings) if item is not None)])
        return embeddings

    for i in range(1, 6):

        print("............START........fold{}.................".format(i))    

        train_files = df.loc[df['fold'] != i, 'filename'].tolist()
        train_labels = df.loc[df['fold'] != i, 'label'].tolist()
        validation_files =  df.loc[df['fold'] == i, 'filename'].tolist()
        validation_labels = df.loc[df['fold'] == i, 'label'].tolist()

        train_ids = [int(i.split('_')[-2]) for i in train_files]
        train_embeddings = clean_embeddings([pd_num_cat[pd_num_cat['Nodule_ID'] == i][['Age', 'PackYears', 'Gender_1', 'Gender_2',
       'SmokingStatus_1', 'SmokingStatus_2', 'SmokingStatus_3']].values for i in train_ids])

        val_ids = [int(i.split('_')[-2]) for i in validation_files]
        val_embeddings = clean_embeddings([pd_num_cat[pd_num_cat['Nodule_ID'] == i][['Age', 'PackYears', 'Gender_1', 'Gender_2',
       'SmokingStatus_1', 'SmokingStatus_2','SmokingStatus_3']].values for i in val_ids])

        train_data = [{"image": img, "label": label,"embedding": emb} for img, label, emb in zip(train_files, train_labels, train_embeddings)]
        validation_data = [{"image": img, "label": label,"embedding": emb} for img, label, emb in zip(validation_files, validation_labels, val_embeddings)]
        
        train_transforms = Compose([
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            RandRotated(keys=keys, prob=0.6, range_x=[0.4,0.4]),  # prob = 0.6, 0.8(X), 1(X)
            RandFlipd(keys=keys, prob=0.6),
            Resized(keys=keys, spatial_size=(32, 32, 32), mode = "trilinear"),
            ToTensord(keys=keys),
        ])

        validation_transforms = Compose([
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Resized(keys=keys, spatial_size=(32, 32, 32), mode = "trilinear"),
            ToTensord(keys=keys),
        ])

        train_dataset = Dataset(data=train_data, transform=train_transforms)
        validation_dataset = Dataset(data=validation_data, transform=validation_transforms)

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=4, shuffle=False)

        model = ResNet18_merge().to(device)
        # print(model)
        
        model_name = 'your model name'
        # print(model_name)
                                            
        class_weights = torch.tensor([1.0 / 706, 1.0 / 134]).to(device)

        loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)

        optimizer = torch.optim.Adam(model.parameters(), 1e-4)
        
        # start a typical PyTorch training
        best_metric = -1

    #   writer = SummaryWriter()
        max_epochs = 5
        
        # Create dictionaries for dataloaders, losses, and accuracies
        dataset_sizes = { 'train': len(train_dataset), 'val': len(validation_dataset) }
        dataloaders = {'train': train_loader, 'val': validation_loader}
        
        losses = {'train': [], 'val': []}
        accuracies = {'train': [], 'val': []}
        cm = {'train': [], 'val': []}
        precisions = {'train': [], 'val': []}
        recalls = {'train': [], 'val': []}
        specificity = {'train': [], 'val': []}
        f1_scores = {'train': [], 'val': []}
        val_probabilities = []
        val_true_labels = []
        val_predicted_labels = []

        for epoch in range(max_epochs):
            
            print(f"epoch {epoch + 1}/{max_epochs}")
            
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                true_labels = []
                predicted_labels = []
                epoch_prob = {
                    'nodule_location': [],
                    'prob': [],
                }

                # Iterate over data
                for batch_data in dataloaders[phase]:
                    
                    inputs, labels, embs = batch_data['image'].to(device), batch_data['label'].to(device), batch_data['embedding'].to(device)
                    optimizer.zero_grad()
                    
                    # Set your desired threshold value
                    threshold = 0.5

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        
                        outputs = model(inputs,embs)
                        prob = torch.nn.functional.softmax(outputs, dim=1)[:,1] 
                        preds = (prob > threshold).long() # function 3

                        loss = loss_function(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    true_labels.extend(labels.data.cpu().numpy()) 
                    predicted_labels.extend(preds.cpu().numpy()) 
                    if phase == 'val':
                        # print(prob)
                        for f in range(prob.numel()):  # batch_size
                            epoch_prob['nodule_location'].append(prob.meta['filename_or_obj'][f])
                            epoch_prob['prob'].append(prob.data.tolist()[f])
                    
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                epoch_cm = confusion_matrix(true_labels,predicted_labels)
                epoch_precision = precision_score(true_labels, predicted_labels)
                epoch_recall = recall_score(true_labels, predicted_labels)
                epoch_f1_score = f1_score(true_labels, predicted_labels)
                
                losses[phase].append(epoch_loss)
                accuracies[phase].append(epoch_acc)
                cm[phase].append(epoch_cm)
                precisions[phase].append(epoch_precision)
                recalls[phase].append(epoch_recall)
                f1_scores[phase].append(epoch_f1_score)
                if phase == 'val':
                    val_probabilities.append(epoch_prob)
                    val_true_labels.append(true_labels)
                    val_predicted_labels.append(predicted_labels)

                print(f"{phase}_loss: {epoch_loss:.4f}", 
                      f"{phase}_acc: {epoch_acc:.4f}",
                      f"{phase}_cm:{epoch_cm.reshape(-1)}",
                      f"{phase}_precision: {epoch_precision:.4f}", 
                      f"{phase}_recall/sensitivity: {epoch_recall:.4f}",
                      f"{phase}_specificity: {epoch_cm[0,0]/(epoch_cm[0,0]+epoch_cm[0,1]):.4f}",
                      f"{phase}_f1_score: {epoch_f1_score:.4f}")          
            
        print("............END........fold{}.................".format(i))  
        print()
    
if __name__ == '__main__':      
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    running_date = datetime.datetime.now().strftime("%Y_%m_%d_%H")

    train_val_model(device, running_date)

    print()
    print("finished!!!!")

