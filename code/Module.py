#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math

class Positional_Encoding(nn.Module):
    '''
    Description:
        Positional Encoding for visits and modalities.
    '''
    def __init__(self,d_model,max_seq_len):
        super().__init__( )

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        pe = torch.zeros(self.max_seq_len, self.d_model)
        position = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x,t=None):
        '''
        Inputs:
            x: tensor - Any features or Embeddings of shape (b,s,d)
            m/t: tensor - Any index sequence of (b,s,1) or (b,s)
        Outputs:
            x: modified tensor of shape (b,s,d)
        '''
        if t is not None:
            x = x + self.pe[t.view(-1).long()].view(t.size(0),t.size(1),-1).clone().detach()
        else:
            x = x + self.pe[0:x.size(1)].view(-1,x.size(1),self.d_model).clone().detach()
        return x

        
class Mask(nn.Module):
    '''
    Input: 
        x: modality list or time list  
            (b,s) or (b,s,1)

    Outputs:
        mask: boolean mask matrix broadcastable with attention matrix. 
        (num_samples, 1, 1,s_v) to (num_samples, n_heads,s_q,s_v)

    '''
    def __init__(self,padding=-1):
        super().__init__()
        self.padding = padding
    
    def forward(self,x):
        '''
        Input:
            x: tensor - (b,s) or (b,s,1)
        Output:
            mask: tensor of shape (n_samples, 1,1,s_v) which can be broadcasted
            to shape of (num_samples, n_heads, s_q,s_v) according to the
            attention matrix to be masked.
        '''
        x = x.view(x.size(0),-1)
        mask = (x != self.padding).unsqueeze(-2) # (num_samples, 1, s_v)
        mask = mask.unsqueeze(-3) 
        return mask
        
        
class Scaled_Dot_Product_Attention(nn.Module):
    '''
    Description: 
        The class output an attentioned matrix from the input sequence.
    
    Input: Q, K, V  : tensor (float32)
          # Q, K: (num_samples, n_head, length_Q,d_k); 
          # V: (num_samples, n_head, length_V, d_v)
          # mask: (num_samples, 1,1,  length_V)
                
    Output: Attention 
        # V_new: tensor (num_samples, n_head, s_v, d_v)
        # attention: tensor (num_samples, n_head,s_q, s_v)
    '''
    def __init__(self,scale):
        super().__init__()
        # the default scale is sqrt(d_k) according to
        # the transformer paper
        self.scale = scale
        
        
    def forward(self,Q, K, V, mask=None):
        '''
        Inputs:
            Q: float tensor of shape (n_sample,n_heads, s_q, d_k)
            K: float tensor of shape (n_sample,n_heads, s_v, d_k)
            V: float tensor of shape (n_sample,n_heads, s_v, d_v)
            mask: (n_samples, 1, 1, s_v)
        Outputs:
            V: attentioned V with shape (n_samples,n_heads,s_q,d_v)
        '''
        attention = torch.matmul(Q, K.transpose(2,3)) # (num_samples, n_head, s_q, s_v)
        attention = attention/self.scale

        if mask is not None:
            attention = attention.masked_fill(mask==0,-1e30)
        attention = nn.functional.softmax(attention,dim=-1)  
        V = torch.matmul(attention,V)
        
        return V


class Multi_Head_Attention(nn.Module):
    '''Basic Module for Encoder.
    
    Pipeline:
        Embedding sequence of dimension d_model ->
        Linear Transformation to produce queries, keys and values of ->
        Attentioned values based on these pairs ->
        Transformation of values to output of d_model ->
        LayerNormalization

    '''

    def __init__(self, n_heads,d_model, 
                        dropout=0.0,
                       d_k=None, d_v=None,
                       d_delta=None):
        
        super().__init__()
        # must-have parameters
        self.n_heads = n_heads
        self.d_model = d_model
        
        # parameters with defaults
        self.d_delta=d_delta
        if self.d_delta is None:
            self.d_delta = d_model 

        self.d_k = d_k
        self.d_v = d_v
        if d_k is None:
            self.d_k = math.ceil(d_model/n_heads)
        if d_v is None:
            self.d_v = math.ceil(d_model/n_heads)

        self.W_Q = nn.Linear(self.d_delta,self.n_heads*self.d_k,bias=False)
        self.W_K = nn.Linear(self.d_model,self.n_heads*self.d_k,bias=False)
        self.W_V = nn.Linear(self.d_model,self.n_heads*self.d_v,bias=False)
        self.attention = Scaled_Dot_Product_Attention(scale=math.sqrt(self.d_k))
        self.fc = nn.Linear(self.n_heads*self.d_v,self.d_model,bias=False)
        
        ## added later
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.d_model,eps=1e-6)
        
    def forward(self,Q, K, V,mask=None):
        '''
        Inputs: 
            Q: float tensor of shape (n_sample, s_q, d_delta)
            K: float tensor of shape (n_sample,s_v, d_model)
            V: float tensor of shape (n_sample,s_v, d_model)
        Outputs:
            V: attentioned V with shape (n_samples,s_q,d_model)
        '''
        b = V.size(0)
        s_q = Q.size(1)
        s_k = K.size(1)
        s_v = V.size(1)

        # Q is external memories for queries
        # Q is typically (1,1,d_delta)
        if Q.size(0) != b:
            Q= Q.expand(b,-1,-1)
        residual = V
        Q = self.W_Q(Q).view(b,s_q,self.n_heads,self.d_k).transpose(1,2) 
        K = self.W_K(K).view(b,s_k,self.n_heads,self.d_k).transpose(1,2)
        V = self.W_V(V).view(b,s_v,self.n_heads,self.d_v).transpose(1,2)
        V = self.attention(Q, K, V, mask=mask) 
        V = V.transpose(1,2).contiguous().view(b,s_q,-1)
        V = self.fc(V)  ## where is the activation go
        V = self.dropout(V)
        if V.size(1) == residual.size(1):
            V += residual
        V = self.norm(V)
        return V
 
class Stack(nn.Module):
    '''
    Pipeline:
        Embedding sequence of dimension d_model ->
        Linear Transformation to produce queries, keys and values of ->
        Attentioned values based on these pairs ->
        Transformation of values to output of d_model ->
        LayerNormalization
    '''
    def __init__(self,n_heads,d_model, 
                      d_k=None, d_v=None,d_ff=None,
                      dropout=0.0,d_delta=None):
        
        super().__init__()

        # must-have variables
        self.n_heads = n_heads
        self.d_model = d_model

        # variables with defaults
        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff
        self.d_delta = d_delta
        if self.d_k is None:
            self.d_k = math.ceil(self.d_model/self.n_heads)
        if self.d_v is None:
            self.d_v = math.ceil(self.d_model/self.n_heads)
        if self.d_ff is None:
            self.d_ff =  self.d_model*2
        if self.d_delta is None:
            self.d_delta=self.d_model
        self.dropout = dropout

        self.attention = Multi_Head_Attention(n_heads=self.n_heads,
                                d_model=self.d_model, d_k=self.d_k,d_v=self.d_v,
                                d_delta=self.d_delta)
       
        # Internal Linear Transformation
        self.fc1 = nn.Linear(self.d_model,self.d_ff)
        self.fc2 = nn.Linear(self.d_ff,self.d_model)
        self.dropout= nn.Dropout(self.dropout)
        self.norm = nn.LayerNorm(self.d_model)
         
    def forward(self,Q,K,V,mask=None):

        # Multi-Head Attention Block
        V = self.attention(Q,K,V,mask)
        
        # Feed-forward layers
        residual = V
        V = self.fc2(nn.functional.relu(self.fc1(V)))  
        V = self.dropout(V)
        V += residual
        V = self.norm(V)
        return V

class Input_Transform(nn.Module):
    '''Transformation of the inputs from multiple modalities
    
    This module transforms the input feature of shape (b,s,d_feature) to
    embeddings of uniform sizes (b,s,d_model). The uniform sizes
    are needed because embeddings across modalities need to be merged in 
    the same encoder. 
    '''
    def __init__(self,d_input,hidden_sizes,d_output):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append( nn.Linear(d_input, hidden_sizes[0]))
        self.layers.append(nn.ReLU())

        # This requires the hidden_sizes to be at least two dimensional
        if len(hidden_sizes) > 0:
            for i in range(len(hidden_sizes) - 1):
                self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
                self.layers.append(nn.ReLU())
        
            self.layers.append( nn.Linear(hidden_sizes[-1],d_output))
            self.layers.append(nn.ReLU())

        else:
            self.layers.append( nn.Linear(d_input,d_output))
            self.layers.append(nn.ReLU())
    
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
        

class Encoder(nn.Module):
    '''Encoder for merging a sequence or Transforming a sequence
    
    Description:
            There are two kinds of encoders: Fusing sequence and transforming sequence.
            Hence, the mask here is considered differently for external Q and consistent Q.
            When external Q is present in which Q.size(1) != V.size(1), the encoder is desgined
            to fuse a sequence. The masking is not needed except for basic stack.
    '''
    def __init__(self,n_heads,d_model, 
                      d_k=None, d_v=None,d_ff=None,
                      dropout=0.0,n_enc_lays=1,d_delta=None,
                      **kwargs):
        super().__init__()
        # Must-Have Parameters
        self.n_heads = n_heads
        self.d_model = d_model

        # parameters with defaults
        self.n_enc_lays = n_enc_lays 
        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff
        self.dropout = dropout
        self.d_delta = d_delta
        
        # In order to process external queries, the first stack was built with d_delta
        self.basic_stack = Stack(n_heads,d_model, d_k, d_v,d_ff,dropout,d_delta)
        self.stack_list=nn.ModuleList([Stack(n_heads,d_model, d_k, d_v,d_ff,dropout) 
                                                                                        for _ in range(n_enc_lays-1)])
        

    def forward(self,Q,K,V,mask):
        '''
        Input:
            Q: float tensor of shape (n_sample,s_q, d_model)
            K: float tensor of shape (n_sample,s_v, d_model)
            V: float tensor of shape (n_sample,s_v, d_model)
            mask: (n_samples, 1, 1, s_v)

            These Q, K, V are not internal Q, K, V inside the multi-head
            attention module. 
        Outputs:
            x: attentioned x with shape (n_samples,s_q,d_model)
        '''
        x = self.basic_stack(Q, K, V, mask)
        # check out the class description
        # if Q size is different from V, the sequence is already fused into vector
        if Q.size(1) != V.size(1):
            for stack in self.stack_list:
                x= stack(x, x, x)
        else: 
            for stack in self.stack_list:
                x= stack(x, x, x,mask)
        return x
        
            

    
