"""
Assigment 2, week 3 - Introduction to Focus Areas (Data Science)
author: @maxo
link: github.com/scaramir/ifa-2022
date: 2022-11-08
"""

#-----------Hyperparameters-----------
use_normalize = False
pic_folder_path = './data/fold1/'
learning_rate = 0.001
batch_size = 15
num_epochs = 15
num_classes = 2
num_channels = 3
load_trained_model = False
pretrained = True
reset_classifier_with_custom_layers = True
train_network = False
evaluate_network = False
output_model_path = './models/'
output_model_name = 'model_1'
#----------------------------------


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import torchvision.models as models
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import copy
import time, random

from tqdm import tqdm
from mo_nn_helpers import get_mean_and_std
from mo_nn_helpers import *
#from "./../../Project 1/code/exploratory_analysis_and_classifiers.py" import *
# TODO: __initi__.py file to reuse code from project 1

def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        print('-----WARNING-----\nCUDA not available. Using CPU instead.')
    print('Device set to {}.'.format(device))
    return device
device = get_device()

# set seeds for reproducibility
def set_seeds(device = 'cuda', seed = 1129142087):
    random.seed(seed)
    np.random.seed(seed+1)
    torch.random.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    print('Seeds set to {}.'.format(seed))
    return
set_seeds(device)

data_dir = pic_folder_path
# data_dir = 'C:/Users/.../Project 2/data'

if use_normalize: 
    friendly, condoms = get_mean_and_std(data_dir)

# Data augmentation and normalization for training
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize(224),
        transforms.RandomRotation(degrees=(-40, 40)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ]),
    "test": transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ]),
}
if use_normalize:
    data_transforms["train"].transforms.append(transforms.Normalize(mean=friendly, std=condoms, inplace=True))
    data_transforms["test"].transforms.append(transforms.Normalize(mean=friendly, std=condoms, inplace=True))


# ---------------Data Loader------------------
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                    for x in ["train", "test"]}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                    shuffle=True, num_workers=0)
                    for x in ["train", "test"]}

dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "test"]}
class_names = image_datasets["test"].classes
num_classes = len(class_names)
# --------------------------------------------


def get_model(model_type, load_trained_model, reset_classifier_with_custom_layers, num_classes=num_classes, pretrained=True, device='cuda', input_model_path=None, input_model_name=None):
    print('Loading model...')
    if (load_trained_model) & (input_model_path is not None) & (input_model_name is not None):
        model = load_model(input_model_path, input_model_name).to(device)
        print('Loaded model \'{}\'.'.format(input_model_name))
    else:
        # Load the pretrained model from pytorch
        if model_type == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
        elif model_type == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
        elif model_type == 'resnet101':
            model = models.resnet101(pretrained=pretrained)
        elif model_type == 'vgg16':
            model = models.vgg16(pretrained=pretrained)
        elif model_type == 'vgg16_bn':
            model = models.vgg16_bn(pretrained=pretrained)
        elif model_type == 'vgg19':
            model = models.vgg19(pretrained=pretrained)
        elif model_type == 'vgg19_bn':
            model = models.vgg19_bn(pretrained=pretrained)
        elif model_type == 'densenet121':
            model = models.densenet121(pretrained=pretrained)
        elif model_type == 'inception_v3':
            model = models.inception_v3(pretrained=pretrained)
        elif model_type == 'googlenet':
            model = models.googlenet(pretrained=pretrained)
        elif model_type == 'shufflenet_v2_x0_5':
            model = models.shufflenet_v2_x0_5(pretrained=pretrained)
        elif model_type == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=pretrained)
        elif model_type == 'resnext50_32x4d':
            model = models.resnext50_32x4d(pretrained=pretrained)
        elif model_type == 'wide_resnet50_2':
            model = models.wide_resnet50_2(pretrained=pretrained)
        elif model_type == 'mnasnet0_75':
            model = models.mnasnet0_75(pretrained=pretrained)
        else:
            print('Model type not found.')
            return None

    if reset_classifier_with_custom_layers:
        model.logits = nn.Sequential(nn.Linear(model.final_shape[0], 256),
                                    nn.Dropout(p=0.4, inplace=True),
                                    nn.Linear(256, 100),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(100, num_classes))
    model = model.to(device)
    print("Done.")
    return model
model = get_model(model_type='resnet18', load_trained_model=False, reset_classifier_with_custom_layers=True, num_classes=num_classes, pretrained=True, device='cuda', input_model_path=None, input_model_name=None)

criterion = nn.CrossEntropyLoss()
# SGD optimizer with momentum could lead faster to good results, but Adam is more stable
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
 

model = train_nn(model, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epochs, device=device, dataloaders=dataloaders, dataset_sizes=dataset_sizes, class_names=class_names, save_model=True, save_model_path=output_model_path, save_model_name=output_model_name)


# TODO: Evaluation of 3 different networks. Use Sigmoid and max to get the probabilities for each of the binary classes.
def evaluate_model(model, dataset_sizes, criterion, class_names, image_datasets, device="cuda", dataset = "test"):
    # for every image of our test set, we will prdict the class and the probability
    # save the probabilities and the classes in a list
    # save the ground truth classes in a list
    # calculate the loss and the accuracy
    # plot the confusion matrix with the older heatmap function from project 1. 
    # plot the ROC curve with the function from project 1
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                    shuffle=False, num_workers=0)
                    for x in ["train", "test"]}

    num_samples = 0
    num_correct = 0
    true_labels_list = []
    pred_labels_list = []
    pred_scores_list = []
    file_names_list = []
    model.eval()
    
    if device == "cuda":
        model.cuda.empty_cache()
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(dataloaders[dataset]), desc="Evaluating the model..."):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            scores = torch.sigmoid(outputs)
            scores = torch.max(scores, 1)
            pred_scores, pred_labels = scores

            num_correct = torch.sum(pred_labels == labels.data)
            num_samples += pred_labels.size(0)
            true_labels_list.append(class_names[labels.cpu().detach().numpy()[0]])
            pred_labels_list.append(class_names[pred_labels.cpu().detach().numpy()[0]])
            pred_scores_list.append(pred_scores.cpu().detach().numpy()[0])
            file_names_list.append(image_datasets[dataset].imgs[i][0].split("/")[-1])

    accuracy = num_correct / num_samples
    loss = criterion(outputs, model)
    print("Accuracy: {:.2f} %".format(accuracy * 100))
    print("Loss: {:.2f}".format(loss))

    print(classification_report(true_labels_list, pred_labels_list, target_names=class_names))
    print(confusion_matrix(true_labels_list, pred_labels_list, labels=class_names))
    

    df = pd.DataFrame({
        "file_name": file_names_list,
        "true_label": true_labels_list,
        "pred_label": pred_labels_list,
        "pred_score": pred_scores_list})
    print("Done.")
    return df




# TODO: Plot the results. (Also with a confusion matrix as heatmap?)

# TODO: report? 