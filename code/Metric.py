#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import  roc_auc_score
        
def Accuracy(y,pred,threshold=None,average=None,step=0.1):
    '''
    Inputs:
        y: binary tensor with shape (samples, n_labs), n_labs is 1 if
            only one
        pred: 0 to 1 array as float tensor

    '''
    assert y.ndim == 2 and pred.size() == y.size()
    assert average in [None, "mean"]

    if threshold is not None:
        pred = pred > threshold
        best_accuracy = (y == pred).float().mean(dim=0)

    else:
        best_accuracy = torch.zeros(y.size(1))
        for threshold in torch.arange(0.01, 1.0, 0.01):
            pred_labs = (pred >= threshold).float()
            accuracy = (pred_labs== y).float().mean(dim=0)
            best_accuracy=torch.where(accuracy > best_accuracy,accuracy,best_accuracy)

    if average == "mean":
        best_accuracy = best_accuracy.mean().item()
    else:
        best_accuracy = best_accuracy.numpy()
    return  best_accuracy



def AUC(y,pred,average=None):
    '''AUC calculation for tensors
    Inputs:
        pred: float tensor of (samples,n_labs)
        y: float tensor of (samples,n_labs), has to be binary
    Ouptut:
        roc_score: score for each class is returned.
        (n_labs,) if average is None
    '''
    assert y.ndim == 2 and pred.size() == y.size()
    roc_score = roc_auc_score(y.detach().numpy(),pred.detach().numpy(),average=average)
    return roc_score




def aucs(targs, preds):
    """
    Calculates the AUC (Area Under the ROC Curve) as an accuracy metric.
    
    Args:
        targs: # int tensor, ndim=2,(samples,n_labs)
        preds: # float tensor,ndim=2,(samples,n_labs)
    Returns:
        float: AUC scores.
    """
    targs = targs.numpy()
    preds = preds.numpy()
    assert targs.ndim == 2 and preds.ndim==2, "Dimension error, ndim must be 2"
    aucs = []
    for i in range(preds.shape[-1]):
        aucs.append(roc_auc_score(targs[:,i],preds[:,i]))
    return aucs

def WAUC(model,dataloader,average=None):
    '''Weighted aucs according to the number of visits
    Inputs:
        model
        dataloader
    Outputs:
        aucs: all kinds of aucs
    '''
    ys = None
    preds = None
    ts = None
    
    ## Gather all data into the same batch
    for ibatch, (x,t,m,y,c,ci,_) in enumerate(dataloader):
        outputs = model(x,t,m,c,ci)
        pred = torch.nn.functional.sigmoid(outputs)
        if ibatch == 0:
            ys = y
            preds = pred
            ts = t
        else:
            ys = torch.cat([ys,y],dim=0)
            preds = torch.cat([preds,pred],dim=0)
            if isinstance(t,list):
                for i,it in enumerate(t):
                    ts[i] = torch.cat([ts[i],it],dim=0)
            else:
                ts = torch.cat([ts,t],dim=0)
    if isinstance(ts,list):
        ts = torch.cat(ts,dim=-1)
        ts,_ = ts.max(dim=-1)

    ts=ts.flatten(1)
    ## Get auc for corresponding number of visits
    visits = sorted(torch.unique((ts!=-1).sum(dim=1)).numpy().tolist(),reverse=True)
    aucs={}
    n_labels = ys.size(1)
    n_patients=ys.size(0)
    
    for i in range(n_labels):
        aucs[i]={}
        ys_temp=torch.empty((0, n_labels), dtype=torch.float32)
        preds_temp=torch.empty((0, n_labels), dtype=torch.float32)
        keys=""
        weighted_auc=0
        for ivisits in visits:
            index = torch.where((ts!=-1).sum(dim=1)==ivisits)[0]
            ipred=preds[index]
            iy = ys[index]
            keys += str(ivisits)+","
            ys_temp=torch.cat([ys_temp,iy],dim=0)
            preds_temp=torch.cat([preds_temp,ipred])

            try:
                auc = AUC(ys_temp[:,[i]],preds_temp[:,[i]])
                weighted_auc+=auc*ys_temp.shape[0]
                aucs[i][keys] = round(auc,4)
                # reinitialize
                ys_temp = torch.empty((0, n_labels))
                preds_temp = torch.empty((0, n_labels))
                keys=""
            except:
                continue
        weighted_auc /= n_patients 
        aucs[i]["weighted"]=weighted_auc
    auc_labels = AUC(ys,preds)
    aucs["total_label"] = auc_labels
    return aucs

