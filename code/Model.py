#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
import Module
import Task
import Merge

class MultimodalLongitudinalModel(nn.Module):
    '''Multimodal Longitudinal Model

    '''
    def __init__(self, params):
        '''
        params: nested dictionary
        {   
        "ModalMerge":
        { 
            
            "modality": {modality_name: {"d_feature":, "hidden_sizes":}, ...}  
            "d_model":,  "n_heads":,  # must have
            "n_enc_lays":,                  # better have
            "d_k":, "d_k":,"d_v","padding","max_num_mods","d_delta","dropout"
        },
        "LongitudinalMerge":
        {
            "n_heads","d_model",  # must have
            "n_enc_lays":, # better have
            "d_k"=None, "d_v"=None,"d_ff"=None, # w defaults
            "dropout"=0.0,"d_delta":,"max_seq_len":
        },
        "Task": 
        {
             "num_tasks":, "d_model","output_size," #must-have
             "n_heads":1,"d_delta"=4, # better have
             d_k,d_v,d_ff,hidden_sizes,dropout=0.0 # w/ defaults
        }
        }
        '''
        super().__init__()

        # params
        self.modalMergeParams = params['ModalMerge']
        self.longitudinalMergeParams = params['LongitudinalMerge']
        self.taskParams = params['Task']

        self.num_mods = len(self.modalMergeParams["modality"])
        self.num_tasks = self.taskParams['num_tasks']

        # modules
        self.modalMerge =  Merge.ModalMerge(self.modalMergeParams)
        self.pe = Module.Positional_Encoding(self.longitudinalMergeParams['d_model'],
                                                                      self.longitudinalMergeParams['max_seq_len'])
        # TB Corrected, the padding should use the task parameters.
        self.mask = Module.Mask(self.longitudinalMergeParams['padding'])

        self.longitudinalMerge = Module.Encoder(**self.longitudinalMergeParams)
        self.tasks = Task.Tasks(**self.taskParams)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,x,t,m,covs=None,covs_index=None):
        '''
        x_list: list # [(b,s,d_f_1),(b,s,d_f_2),...]
        t_seq: tensor (b,s,1)
        m_seq: tensor # [b,s,n_mods] 
        covs: tensor # (b,n_covs,1)
        covs_index: tensor # (b,n_covs)
        '''
        # Merge between modalities
        # something to bear in mind that when x are padded to the same length, this
        # step will not mask them. next step will mask them.
        x = self.modalMerge(x,m) # (b,s,d_model)

        # Longitudinal merge
        x = self.pe(x,t) 
        mask = self.mask(t) # t squeeze will not work for b == 1
        x = self.longitudinalMerge(x,x,x,mask) 
        
        if covs_index is None:
            covs_mask= None
        else:
            covs_mask = self.mask(covs_index) # b,1,1n_covs

        # At least one feature set exist from multiple modalities. Mask for x is not needed
        x = self.tasks(x,x_mask=None,covs=covs,covs_mask=covs_mask) #(num_task,output_size)
        return x


def Create_Model(params):
    model = MultimodalLongitudinalModel(params)
    return model

def Save_Model(model,path):
    torch.save(model.state_dict(),path)
    print(f"Model saved to {path}")

def Load_Model(model,path):
    model.load_state_dict(torch.load(path))
    model.eval()  
    return model
