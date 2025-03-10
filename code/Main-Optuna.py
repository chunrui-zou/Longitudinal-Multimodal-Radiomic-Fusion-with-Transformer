#!/usr/bin/env python
# coding: utf-8
import os
import sys
import math
import torch
import Merge
import Model
import Data
import Training
import Loss
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import Parameters
import torch.optim as optim
import numpy as np
import importlib
from torch.optim.lr_scheduler import StepLR
import Metric
import sklearn.metrics as metrics
from importlib import reload
import random
import optuna
device = torch.device("cpu")

################# Load Data ###############
data_path = "path/to/pickled-data.pkl"
lmr = Data.LMR() # should add some function to the initialization
lmr.Load_Pickle(data_path)
features = ["CR_","CT_"]
labels = ['mortality', 'vent', 'icu']
covs = ["age","gender"]
covs_stand = [True, False]
datasets = lmr.GetOutput(features,labels,
                         covs = covs,
                         covs_stand=covs_stand,format="ML")

# split data
splits_ind = Data.data_split(datasets,which_y=0,ratio=0.8)
train_ind = []
test_ind = []
for iv in splits_ind.keys():
    for ilab in splits_ind[iv].keys():
        train_ind+=splits_ind[iv][ilab][0]
        test_ind+=splits_ind[iv][ilab][1]
test_set=torch.utils.data.Subset(datasets,test_ind)

for _ in range(5):
    random.shuffle(train_ind)
remaining_set = torch.utils.data.Subset(datasets,train_ind)

train_size = math.floor(len(train_ind) * 0.8)
val_size = len(train_ind) - train_size
train_set, val_set= random_split(remaining_set, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=500, shuffle=True)
val_loader = DataLoader(val_set, batch_size=500,shuffle=True)
test_loader = DataLoader(test_set, batch_size=500, shuffle=True)

# Create Model
best_model = None
best_auc = 0.0
def objective(trial):
    global best_model
    global best_auc
    model = Model.Create_Model(Parameters.ML)
    model.to(device)
    lr_op = trial.suggest_float('learning rate', 0.00000005,0.000001)
    decay_op = trial.suggest_float('decay', 0.001,0.1)
    pw_1 = trial.suggest_float('pos_weight 1', 1.0,6.0)
    pw_2 = trial.suggest_float('pos_weight 2', 1.0,6.0)
    pw_3 = trial.suggest_float('pos_weight 3', 1.0,6.0)
    lw_1 = trial.suggest_float('lab_weight 1', 0.5,2)
    lw_2 = trial.suggest_float('lab_weight 2', 0.5,2)
    lw_3 = trial.suggest_float('lab_weight 3', 0.5,2)
    
    optimizer = optim.Adam(model.parameters(), lr=lr_op,
                                            weight_decay=decay_op)
    label_weight = torch.tensor([lw_1,lw_2,lw_3])
    pos_weight = torch.tensor([pw_1,pw_2,pw_3])

    # Train Model
    trainer = Training.Trainer(model=model,device=device,nfeatures=2)
    trainer.Train(train_loader=train_loader,loss_fn = Loss.Multilabel_Loss,
          optimizer=optimizer,epochs=8000,val_loader=val_loader, 
          pos_weight= pos_weight,
          label_weight=label_weight,
          patience=100,loss2Watch=which_y
                 )        
   
    auc2Watch = Metric.WAUC(model,test_loader)[which_y]['weighted']
    auc2Record = Metric.WAUC(model,test_loader)
    trial.set_user_attr("auc2Record", auc2Record)

    if auc2Watch > best_auc:
        best_auc = auc2Watch
        best_model = model

    return auc2Watch

study = optuna.create_study(direction="maximize", 
                            sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=100)
best_trial = study.best_trial

print("Best Trial:")
for key, value in best_trial.params.items():
        print("{}: {}".format(key, value))
print("Best Trial AUCs:")
print(best_trial.user_attrs["auc2Record"])
Model.Save_Model(best_model,"best_model.pth")
