#!/bin/python
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset,random_split
import Model
import Parameters
import Optim
import Loss
import Metric

class Trainer(nn.Module):
    
    def __init__(self, model,device="cpu",nfeatures=2):
        super().__init__()
        self.model = model
        self.device = device
        self.nfeatures=nfeatures
        self.avgLoss_epochs = []
        self.loss_epochs= [] 
        self.avgValLoss_epochs = []
        self.valLoss_epochs = []

    def Evaluate(self, dataloader, loss_fn,pos_weight=None,label_weight=None):
        self.model.to(self.device)
        self.model.eval()
        running_loss = 0.0
        running_loss_labs = torch.tensor([0.0,0.0,0.0])
        running_accuracy = torch.tensor([0.0])
        running_accuracy_labs = torch.tensor([0.0,0.0,0.0])

        with torch.no_grad():
            for x, t, m, y, c, ci,_ in dataloader:
            
                # Forward pass
                outputs = self.model(x,t,m,c,ci)
                loss, loss_labs = loss_fn(targets=y,predictions=outputs,pos_weight=pos_weight,label_weight=label_weight)
                running_loss += loss
                running_loss_labs += loss_labs

                # Compute accuracy
                predicted = nn.functional.sigmoid(outputs)
                accuracy = Metric.Accuracy(y,predicted,average="mean")
                running_accuracy += accuracy
                accuracy_labs =  Metric.Accuracy(y,predicted)
                running_accuracy_labs += accuracy_labs
              

        average_loss = running_loss / len(dataloader)
        average_loss_labs = running_loss_labs / len(dataloader)
        average_acc_labs = running_accuracy_labs/len(dataloader)
        average_acc = running_accuracy/len(dataloader)
        
        return average_loss, average_loss_labs, average_acc, average_acc_labs


    def Train(self,train_loader,loss_fn,optimizer,
                    epochs,
                    val_loader=None,pos_weight=None,
                    label_weight=None,writer=None,patience=None,loss2Watch=0):
        '''
        Everything here should be kept tensors.
        patience: after patiences, the loss2watch loss is not improving , early stop
        '''
        self.model.train()
        self.model.to(self.device)
        loss_epochs = []
        avgLoss_epochs = []
        valLoss_epochs = []
        avgValLoss_epochs = []
        valAcc_epochs = []
        avgValAcc_epochs = []
        moving_patience = patience
        bestValLoss = 100000
        print("Start training ......")
        print("=================")
        print("")
        for iepoch in range(epochs):
            print(f"Epoch: {iepoch}")

            running_avgLoss = torch.tensor(0.0)
            running_loss = torch.tensor([0.0,0.0,0.0])

            for ibatch, (x,t,m,y,c,ci,_) in enumerate(train_loader):
                
                outputs = self.model(x,t,m,c,ci)
                avgLoss_ibatch,loss_ibatch = loss_fn(targets=y,predictions=outputs,
                                                                            pos_weight=pos_weight,
                                                                            label_weight=label_weight)

                # the loss can be controlled with weights

                avgLoss_ibatch.backward()
                optimizer.step()

                running_avgLoss += avgLoss_ibatch
                running_loss += loss_ibatch

                if writer is not None:
                    for name, param in self.model.named_parameters():
                        writer.add_histogram(f"{name}", param, iepoch)
                        writer.add_histogram(f"{name}_grad", param.grad, iepoch)
                    

            loss_iepoch = running_loss/len(train_loader)
            avgLoss_iepoch = running_avgLoss/len(train_loader)
            self.loss_epochs.append(loss_iepoch.detach().numpy())
            self.avgLoss_epochs.append(avgLoss_iepoch.detach().item())
            
            # print results
            print(f"Train Loss: {loss_iepoch.detach().numpy()} | {avgLoss_iepoch.detach().item()}")
            
            # validation loss
            if val_loader is not None:
                avgValLoss_iepoch, valLoss_iepoch, valAcc_iepoch, avgValAcc_iepoch =  self.Evaluate(val_loader,loss_fn,
                                                                                                                pos_weight=pos_weight,label_weight=label_weight)
                self.avgValLoss_epochs.append(avgValLoss_iepoch.detach().item())
                self.valLoss_epochs.append(valLoss_iepoch.detach().numpy())

                print(f"Val Loss: {valLoss_iepoch.detach().numpy()} | {avgValLoss_iepoch.detach().item()}")
                
                if patience is not None:
                    #valLoss2Watch = valLoss_iepoch.detach().numpy()[loss2Watch]
                    valLoss2Watch =  avgValLoss_iepoch.detach().item()
                    if bestValLoss > valLoss2Watch:
                        bestValLoss = valLoss2Watch
                        moving_patience = patience
                    else:
                        moving_patience-=1
                    if moving_patience < 0:
                        return None
            print("")
        return None

    def Get_Losses(self):
        return np.array(self.avgLoss_epochs).reshape(-1,1),np.array(self.loss_epochs), np.array(self.avgValLoss_epochs).reshape(-1,1), np.array(self.valLoss_epochs)
    
    




