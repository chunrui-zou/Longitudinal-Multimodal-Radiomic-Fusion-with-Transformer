#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
import Module


class LongitudinalMerge(nn.Module):
        '''
        The class is designed to merge the longitudinal features within the same modality.
        '''
        def __init__(self,modality,
                                d_input,d_model,
                                n_heads=3,n_enc_lays=3,
                                n_heads_delta = 3, n_enc_lays_delta=1,
                                max_seq_len=200,padding=-1,
                                dropout=0.0,
                                hidden_sizes = None,
                                d_k = None, d_v = None,d_ff=None,d_delta=None
                        ):
            super().__init__()

            # must-have parameters
            self.modality=modality
            self.d_input = d_input
            self.d_model = d_model

            # parameters for longitudinal transformer
            self.n_heads = n_heads
            self.n_enc_lays = n_enc_lays

            # parameters for merger
            self.n_heads_delta = n_heads_delta
            self.n_enc_lays_delta = n_enc_lays_delta
            
            self.padding = padding
            self.max_seq_len = max_seq_len
            self.dropout = dropout

            # parameters able to be set as None
            self.d_k = d_k
            self.d_v = d_v
            self.d_ff = d_ff
            self.hidden_sizes = hidden_sizes
            self.d_delta = d_delta


            # control if final representation is generated
            self._merge = True

            # submodules and tensors
            self.mask = Module.Mask(self.padding)

            # Layers to transform the input features to d_model
            
            if self.hidden_sizes is None:
                self.hidden_sizes = []
                self.hidden_sizes.append(math.ceil(self.d_input/2))
                self.hidden_sizes.append(math.ceil((self.d_input/2 + self.d_model)/2))
            self.fcs = nn.ModuleList()
            self.fcs.append(nn.Linear(self.d_input,self.hidden_sizes[0]))
            self.fcs.append(nn.ReLU())
            for i in range(len(self.hidden_sizes)-1):
                self.fcs.append(nn.Linear(self.hidden_sizes[i],self.hidden_sizes[i+1]))
                self.fcs.append(nn.ReLU())
            self.fcs.append(nn.Linear(self.hidden_sizes[-1],self.d_model))
            self.fcs.append(nn.ReLU())

            self.pe= Module.Positional_Encoding(d_model=self.d_model,
                                                                        max_seq_len=self.max_seq_len)
            
            # Encoder for transforming longitudinal sequences
            self.encoder=Module.Encoder(n_heads=self.n_heads,
                                                d_model=self.d_model,
                                                n_enc_lays=self.n_enc_lays,
                                                d_k=self.d_k,d_v=self.d_v,d_ff=self.d_ff,
                                                dropout=self.dropout)

            self.delta = nn.Parameter(torch.ones(1,1,self.d_delta)) 
            self.delta.requires_grad = True
            self.encoder_delta= Module.Encoder(n_heads=self.n_heads_delta,
                                        d_model=self.d_model,
                                        d_k=self.d_k,d_v=self.d_v,d_ff=self.d_ff,
                                        dropout=self.dropout,
                                        n_enc_lays=self.n_enc_lays_delta,
                                        delta=self.d_delta) 
            

        def turn_on_merge(self):
            self._merge = True
            
        def turn_off_merge(self):
            self._merge= False
        
        def freeze_pretraining(self):
            '''
            freeze_prertaining will freeze the parameters in modules before encoder_delta
            since the pretraining only needs the modules before encoder_delta
            '''
            for layer in self.fcs:
                for param in self.layer.parameters:
                    para.requires_grad = False

            for param in self.encoder.parameters:
                param.requires_grad = False
            
        def unfreeze_pretraining(self):
            for layer in self.fcs:
                for param in self.layer.parameters:
                    para.requires_grad = True
            for param in self.encoder.parameters:
                param.requires_grad = True
                
        def forward(self,x,t):
            '''
            Input: 
                x: tensor # (b,s,d_input)
                t: tensor # (b,s) or (b,s,1)
            Output: 
                if not merge:
                    x: tensor # (b,s,d_model)
                else:
                    x: tensor # (b,1,d_model)
            '''
            
            for layer in self.fcs:
                x = layer(x)
            x = self.pe(x,t)
            mask = self.mask(t.squeeze()) 
            x = self.encoder(x,x,x,mask)

            if self.merge:
                x = self.encoder_delta(self.delta,x,x,mask)
            return x



class ModalMerge(nn.Module):

    def __init__(self,params):
        '''
        params: dict
        { 
        # modality-specific parameters, hidden_sizes are the sizes for hidden layers
        # transforming d_feature to d_model.
        "modality": {modality_name: {"d_feature":, "hidden_sizes":}, ...}  
        "d_model":,
        "n_heads":,
        "n_enc_lays":,
        "d_k":, "d_k":,"d_v","padding","max_num_mods","d_delta","dropout"
        }
        '''
        super().__init__()

        # parameters
        self.num_mods = len(params["modality"])
        self.mod_names = list(params["modality"].keys())
        self.n_heads = params["n_heads"]
        self.n_enc_lays = params["n_enc_lays"]
        self.n_heads_delta = params["n_heads_delta"]
        self.n_enc_lays_delta = params["n_enc_lays_delta"]
        self.d_model = params['d_model']
        self.d_k = params['d_k']
        self.d_v = params['d_v']
        self.d_ff = params['d_ff']
        self.dropout = params['dropout']
        self.padding = params['padding']
        self.max_num_mods = params['max_num_mods']
        self.d_delta = params['d_delta']



        # layers to transform the input to the same embedding size
        self.input_transforms = nn.ModuleList()
        for name in self.mod_names:
            d_feature = params['modality'][name]['d_feature']
            hidden_sizes = params['modality'][name]['hidden_sizes']

            # The input transform does not necessitate hidden layers.
            # The hidden_sizes can be left as None
            if hidden_sizes is None: 
                hidden_sizes = []
                hidden_sizes.append(math.ceil(d_feature/2))
                hidden_sizes.append(math.ceil((d_feature/2 + self.d_model)/2))
            transform = Module.Input_Transform(d_input=d_feature,
                                         hidden_sizes=hidden_sizes,d_output=self.d_model)
            self.input_transforms.append(transform)

        # Encoder to transform the embeddings. 
        self.mask = Module.Mask(self.padding)
        self.pe = Module.Positional_Encoding(self.d_model,self.max_num_mods)    
        self.encoder = Module.Encoder(n_heads =self.n_heads,
                                              d_model=self.d_model, 
                                              d_k = self.d_k,
                                              d_v = self.d_v,
                                              d_ff = self.d_ff,
                                              dropout=0.0,
                                              n_enc_lays=self.n_enc_lays)
        
        # encoder to merge the emebddings
        self.delta = nn.Parameter(torch.ones(1,1,self.d_delta)) 
        self.delta.requires_grad = True

        # This one does not need to be 3 layers, only one layer is fine
        self.encoder_delta = Module.Encoder(n_heads = self.n_heads_delta,
                                                        d_model = self.d_model,
                                                        d_k = self.d_k,
                                                        d_v = self.d_v,
                                                        d_ff = self.d_ff,
                                                        dropout = self.dropout,
                                                        n_enc_lays = self.n_enc_lays_delta, 
                                                        d_delta = self.d_delta)

    def forward(self,x_list,m_seq):
        '''
        x_list: list of tensors
                tensor_mod_1: (b,s,d_feature1)
                tensor_mod_2: (b,s,d_feature2)
                ...
                tensor_mod_m: (b,s,d_featurem)
        m_seq: tensor of shape (b,s,n_mods)
        '''
        b = x_list[0].size(0)
        s = x_list[0].size(1)

        # transform each input to the same d_model size
        x = []
        for i, transforms in enumerate(self.input_transforms):
            x.append(transforms(x_list[i]).unsqueeze(2)) # (b,s,1,d_model)  4 concat
        x = torch.cat(x,dim=2).view(-1,self.num_mods,self.d_model)# (b*s,n_model,d_model)
        
        # Encode multi-modal inputs
        x = self.pe(x,m_seq.view(-1,self.num_mods))
        mask = self.mask(m_seq.view(-1,self.num_mods)) # (b*s,n_mods) -> (b*s,1,1,n_mods)
        x = self.encoder(x,x,x,mask)

        # Merge modalities
        # (b*s,n_mods,d_model) --> (b*s,1,d_model) -> (b,s,d_model)
        x = self.encoder_delta(self.delta,x,x,mask).view(b,s,self.d_model)
        return x
        
        
