#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
import Module

class Delta_Generation(nn.Module):
    
    '''Generate the External Memory for Single Task
    
    Input:
        d_model: int -- the dimension of external memory
        n_heads: int

    Output: 
        V: delta in single task
        The generated delta is supposed to extract: 
        information independent from covariates and information relevant to covariates.
    '''
    def __init__(self,d_model = 4,  n_heads=1):

        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # embedding layer
        self.pe = Module.Positional_Encoding(2, 100) 
        self.embedding_lay = nn.Linear(1,2)

        self.d_k = 2
        self.d_v = self.d_k
        self.Q = nn.Parameter(torch.ones(1,1,1))
        self.W_Q = nn.Linear(1,2,bias=False)
        self.W_K = nn.Linear(2,2,bias=False)
        self.W_V = nn.Linear(2,2,bias=False)
        self.attention = Module.Scaled_Dot_Product_Attention(self.d_k)
        self.fc = nn.Linear(2,self.d_model,bias=False)
        self.register_buffer('prefix',  torch.ones((1,1,1)))

    def forward(self,covs=None,mask=None):
        '''
        Input:
            convs: tensor -- (b,num_covs,1) to resemble a sequence
            mask: tensor  -- (b,n_heads,1,num_covs)
        Output:
            V: delta -- external memory of shape (b,1,d_delta) or (1,1,d_delta)
        '''
        
        # Adding prefix to extract info. independent from covs.
        if covs is None:
            covs = self.prefix
            b = 1
            s = 1
        else:
            b = covs.size(0)  ## covs b,s,1 --> b,s,n_heads*d_k
            s = covs.size(1)+1
            covs = torch.cat((self.prefix.expand(b,-1,-1),covs),dim=1).float()

        # Make model covariate-aware by Positional Encoding
        emb_covs = self.embedding_lay(covs) 
        emb_covs = self.pe(emb_covs) 

        # Merging covs into delta by attention
        Q = self.W_Q(self.Q).view(1,1,self.n_heads,self.d_k).transpose(1,2) 
        K = self.W_K(emb_covs).view(b,s,1,2).transpose(1,2)
        V = self.W_V(emb_covs).view(b,s,1,2).transpose(1,2)
        if mask is not None: # mask (b,heads,1,1)
            mask_prefix = torch.ones(mask.size(0),mask.size(1),mask.size(2),1)
            mask = torch.cat((mask_prefix,mask),dim=-1)
        V = self.attention(Q, K, V, mask=mask)  ## (b,n_heads,1,d_v) 
        V = V.transpose(1,2).contiguous().view(b,1,-1)
        V = self.fc(V) # (b,1,d_model) or (1,1,d_model)
        return V

class Task(nn.Module):
    def __init__(self,d_model,output_size, n_heads = 1, n_enc_lays = 1,
		                        hidden_sizes = None,
		                        d_delta=4, d_k=None,d_v=None,d_ff=None,
		                        dropout=0.0):

        super().__init__()
        
        # Must have parameters
        self.d_model = d_model
        self.output_size = output_size

        # parameters with defaults or compatible with None
        self.n_heads = n_heads
        self.n_enc_lays = n_enc_lays

        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff
        self.dropout = dropout
        self.d_delta = d_delta

        # Hidden layers default to a two-layer MLP
        self.hidden_sizes = hidden_sizes
        if self.hidden_sizes is None:
            self.hidden_sizes = []
            self.hidden_sizes.append(math.ceil(self.d_model/2))
            self.hidden_sizes.append(math.ceil((self.d_model/2 + self.output_size)/2))
        assert len(self.hidden_sizes) > 1

        # create delta with and without covariates
        self.delta_gen =  Delta_Generation(d_model=self.d_delta) # (b,1,d_delta)
        
        # merge the embedding sequence into a final representation
        self.pe = Module.Positional_Encoding(self.d_model,100)
        self.encoder_delta= Module.Encoder(n_heads=self.n_heads,
                                                                    d_model=self.d_model,
								                                    d_k=self.d_k,d_v=self.d_v,d_ff=self.d_ff,
								                                    dropout=self.dropout,
                                                                    n_enc_lays=self.n_enc_lays,
								                                    d_delta=self.d_delta)

        # tranform the embeddings
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.d_model, self.hidden_sizes[0]))
        self.layers.append(nn.ReLU())
        for i in range(len(self.hidden_sizes) - 1):
            self.layers.append(nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(self.hidden_sizes[-1],self.output_size))


    def forward(self,x,x_mask=None,covs=None,covs_mask=None):
       
        delta = self.delta_gen(covs=covs,mask=covs_mask)
        x = self.pe(x)  # x is the d_model embedding from fusion
        x = self.encoder_delta(Q=delta,K=x,V=x,mask=x_mask) # (b,1,d_model)
        x = x.squeeze(-2)
        for layer in self.layers:
            x = layer(x)
        return x


class Tasks(nn.Module):
    '''Modules for multi-task learning

    Must-have parameters without defaults:
        num_tasks
        d_model
        output_size

    Parameters better to have:
    	d_delta: the dimensions is better to be set to control the parameters. 

    parameters with defaults but helpful to set it up:
        n_heads: default 1
        n_enc_lays: default 1
    '''

    def __init__(self,num_tasks,d_model,output_size,
			                    n_heads=1,n_enc_lays=1,
			                    d_delta=4,d_k=None,d_v=None,d_ff=None,
			                    hidden_sizes=None,dropout=0.0):

        super().__init__()


        self.tasks = nn.ModuleList([Task(d_model=d_model,
        					output_size = output_size,n_heads = n_heads, 
        					n_enc_lays = n_enc_lays, hidden_sizes = hidden_sizes,
        					d_delta = d_delta, d_k=d_k,d_v=d_v, d_ff = d_ff,
        					dropout = dropout) for _ in range(num_tasks)])

    def freeze_params(self):
        for param in self.parameters:
            param.requires_grad = False
        
    def unfreeze_params(self):
        for param in self.parameters:
            param.requires_grad = True

    def forward(self,x,x_mask,covs=None, covs_mask=None):
        '''
        Inputs:
        	x: embedding sequence, (b,s_x,d)
        	x_mask: masking for sequence, (_,1,s_x)
        	covs: covariates, (b,s_cov,1)
        	covs_mask: masking for covs, (b,1,s_cov)
        Outputs:
        	outputs: the list of outputs from each task, [output_size,output_size,...]
        '''
        outputs = []
        for task in self.tasks:
            outputs.append(task(x=x,x_mask=x_mask,covs=covs,covs_mask=covs_mask))
        outputs = torch.cat(outputs,-1) # b, 1
        return outputs
    

