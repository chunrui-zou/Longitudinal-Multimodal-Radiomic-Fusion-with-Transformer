import torch
from torch.utils.data import TensorDataset
import numpy as np
import pickle
import math

def Index_Tensors(data,index):

    assert torch.is_tensor(index),"Index input must be a torch tensor"
    assert isinstance(data,list) or torch.is_tensor(data), "data is a list of tensors or a tensor"
    if isinstance(data,list):
        output = []
        for idata in data:
           output.append(idata[index])
    else:
        output = data[index]
    return output

class LMR:

    def __init__(self):
        self._raw_data = None

    def Load_Pickle(self, data_path):
        with open(data_path,'rb') as f:
            self._raw_data = pickle.load(f)

        self.x_dict = self._raw_data['x_dict']
        self.t_dict = self._raw_data['t_dict']
        self.y_dict= self._raw_data['y_dict']
        self.c_dict=self._raw_data['c_dict']

        self.nvisits = list(self.x_dict.keys())
        self.max_nvisit = max(self.x_dict.keys())
        self.total_patients = 0
        for invisit in self.x_dict:
            self.total_patients += next(iter(self.x_dict[invisit].items()))[1].shape[0]

    def Down_Passing(self):
        features = list(self.x_dict[self.nvisits[0]].keys())
        labels = list(self.y_dict[self.nvisits[0]].keys())
        covs =  list(self.c_dict[self.nvisits[0]].keys())
        nvisits_sorted = sorted(self.nvisits)
        nvisits_n = len(nvisits_sorted)
        self.total_patients = 0
        for i in range(nvisits_n):
            ivisits = nvisits_sorted[i]
            for j in range(i+1, nvisits_n):
                jvisits = nvisits_sorted[j]
                for ifeature in features:
                    
                    self.x_dict[ivisits][ifeature] = np.concatenate([self.x_dict[ivisits][ifeature], 
                                                        self.x_dict[jvisits][ifeature][:,0:ivisits]],axis=0)
                    self.t_dict[ivisits][ifeature] = np.concatenate([self.t_dict[ivisits][ifeature], 
                                                        self.t_dict[jvisits][ifeature][:,0:ivisits]],axis=0)
 
                for ilabel in labels:
                    self.y_dict[ivisits][ilabel] = np.concatenate([self.y_dict[ivisits][ilabel], 
                                                                                       self.y_dict[jvisits][ilabel]],axis=0)
                for ic in covs:
                    self.c_dict[ivisits][ic] = np.concatenate([self.c_dict[ivisits][ic], 
                                                                                             self.c_dict[jvisits][ic]],axis=0)
            
            self.total_patients += next(iter(self.x_dict[ivisits].items()))[1].shape[0]    

    def _Convert2Tensor(self,x):
        """
        Converts a NumPy array or a list of NumPy arrays to a tensor.

        Args:
            numpy_array: The NumPy array or list of NumPy arrays to convert to a tensor.

        Returns:
            A PyTorch tensor.
        """
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        elif isinstance(x, list):
            return [torch.from_numpy(ix) for ix in x]
        else:
            raise TypeError("numpy_array must be a NumPy array or a list of NumPy arrays")


    def GetFeatures(self,features):
        '''
        Get Padded Features as Tensors
        '''
        assert self._raw_data is not None, "Raw data is not loaded!"
        output = []
        for i, feature in enumerate(features):
            x = []
            for invisit in self.nvisits:
                x_i = self.x_dict[invisit][feature]
                if invisit < self.max_nvisit:
                    npatients = x_i.shape[0]
                    padding = np.zeros((npatients, self.max_nvisit - invisit,x_i.shape[-1]))
                    x_i = np.concatenate([x_i,padding],axis=1)
                x.append(x_i)
            output.append(np.concatenate(x,axis=0))
        return output

    def GetTimes(self,features,format="LM"):
        '''
        Time sequence should have different formats for LM and ML. Oh,my GOD!!!
        Args:
            features (list): List of feature sets
            format (str) : Format of the time sequences (default: LM)
        Output:
            output:Time sequences for each feature set of shape 
            [(n, max_nvisits, 1), ...].
        '''
        assert self._raw_data is not None, "Raw data is not loaded!"
        output = []
        for i, feature in enumerate(features):
            t= []
            for invisit in self.nvisits:
                t_i =self.t_dict[invisit][feature]
                if invisit < self.max_nvisit:
                    npatients = t_i.shape[0]
                    padding = np.full((npatients,self.max_nvisit - invisit,1),-1)
                    t_i = np.concatenate([t_i,padding],axis=1)
                t.append(t_i)
            output.append(np.concatenate(t,axis=0))
        if format=="ML":
            output = np.maximum(*output)
        return output

    def GetOutcomes(self,labels):
        assert self._raw_data is not None, "Raw data is not loaded!"
        output_y = []
        for i, label in enumerate(labels):
            y_label = []
            for invisit in self.nvisits:
                y_i = self.y_dict[invisit][label] # (n,1)
                y_label.append(y_i)
            output_y.append(np.concatenate(y_label,axis=0))
        output_y = np.concatenate(output_y,axis=1)    
        return output_y

    def GetCovs(self,covs,standardize=None):
        assert self._raw_data is not None, "Raw data is not loaded!"
        output_covs= []
        for i, cov in enumerate(covs):
            cov_label = []
            for invisit in self.nvisits:
                cov_i = self.c_dict[invisit][cov] # (n,1)
                cov_label.append(cov_i)
            cov_label = np.expand_dims(np.concatenate(cov_label,axis=0),-1)

            if standardize is not None and standardize[i] == True:
                cov_label = (cov_label-np.nanmean(cov_label))/(np.nanstd(cov_label))

            output_covs.append(cov_label)

        output_covs = np.concatenate(output_covs,axis=1)  
        return output_covs  

    def GetFeaturePositionalIndex(self,features,format="LM"):
        '''
        Output:
            m (tensor): modality index
        '''
        assert self._raw_data is not None, "Raw data is not loaded!"
        total_patients = self.total_patients
        n_features = len(features)
        t = self.GetTimes(features)
        if format=="LM":
            output_m = []
            for i, feature in enumerate(features):
                m = np.full((self.total_patients,1),i+1)
                m[np.all(t[i]==-1,axis=1)] = -1  ## (b,1)
                output_m.append(m)
            output_m = np.concatenate(output_m,axis=1) #(b,n_modalities)
        else:
            # ML, m should be (b,s,n_mods)
            output_m = []
            for i,feature in enumerate(features):
                m = t[i]
                m[m!=-1] = i + 1
                output_m.append(m)
            output_m = np.concatenate(output_m,axis=2)
        return output_m

    def GetPositionalIndex(self,sequence):
        '''
        Description:
            when the sequence contains np.nan values, the corresponding
            index is set to -1.
        Input:
            sequence (array): any feature sequence of shape 
            (n,n_features,d_features)
        Output:
            indexes (array): (n,n_features,1)
        '''
        output = []
        n = sequence.shape[0]
        s = sequence.shape[1]
        ind = np.full((n,s,1),1)
        ind[np.all(np.isnan(sequence),axis=-1)] = -1 # the last dimension is for feature dimensions
        return ind
    
    def GetOutput(self,features,labels,covs=None,covs_stand=None,format="LM",index=None):
        x=self.GetFeatures(features)
        y=self.GetOutcomes(labels)
        t = self.GetTimes(features,format=format) # may be a list or tensor
        m = self.GetFeaturePositionalIndex(features, format=format)
        if covs is not None:
            c=self.GetCovs(covs,standardize=covs_stand)  # (b,n_covs,1)
            # T.B.D  when covs is nan, the
            c_index = self.GetPositionalIndex(c)

        else:
            c = None
            c_index = None

        # Turn everything into a tensor
        x = self._Convert2Tensor(x)
        y = self._Convert2Tensor(y)
        t = self._Convert2Tensor(t)
        m= self._Convert2Tensor(m)
        if c is not None:
            c = self._Convert2Tensor(c)
            c_index = self._Convert2Tensor(c_index)

        if index is not None:
            x = Index_Tensors(x,index)
            t = Index_Tensors(t,index)
            m = Index_Tensors(m,index)
            y = Index_Tensors(y,index)
            c = Index_Tensors(c,index)
            c_index = Index_Tensors(c_index,index)

        datasets = CustomTensorDataset(x,t,m,y,c,c_index)
        #datasets = datasets.to(device)
        return datasets


class CustomTensorDataset(torch.utils.data.Dataset):

    def __init__(self, x,t,m,y,c=None,c_index=None):
        self.x = x
        self.t = t
        self.m = m
        self.y = y
        self.c = c
        self.c_index = c_index
        self.n_visits = None
        self.total_patients = self.y.size(0)

        self._calculate_freqs()
        self._calculate_weights()
        #self._calculate_visits()


    def _calculate_freqs(self):

        def get_frequencies(tensor):
            frequencies = {}
            for value in tensor.unique():
                frequencies[value.item()] = (tensor == value).sum().item()
            return frequencies

        # within each label, positive vs negative
        self.y_freqs = {}
        for i in range(self.y.size(1)):
            freq = get_frequencies(self.y[:,i])
            self.y_freqs[i] = freq

        self.m_freqs=get_frequencies(self.m)

        self.t_freqs = {}
        if isinstance(self.t,list):
            for i, it in enumerate(self.t):
                self.t_freqs[i]= get_frequencies((it !=-1).view(it.size(0),-1).sum(dim=-1))
        else:
            # (b,1) for t_freqs
            self.t_freqs= get_frequencies((self.t !=-1).view(self.t.size(0),-1).sum(dim=-1))

    def _calculate_weights(self):
        '''
        Function to calculate pos_weights between labels and t_weights between samples.
        '''
        self.pos_weights = []
        for ikey in self.y_freqs:
            self.pos_weights.append(self.y_freqs[ikey][0]/self.y_freqs[ikey][1])
        self.t_weights=torch.zeros_like(self.y).float()
        if isinstance(self.t,list):
            for i in range(self.total_patients):
                iweight = 0
                nm = 0
                for im in self.t_freqs:
                    nm+=1
                    freq = (self.t[im][i]!=-1).sum().item()
                    reverse_ratio = self.total_patients/self.t_freqs[im][freq]
                    #reverse_ratio = self.t_freqs[im][freq]/self.total_patients
                    iweight += reverse_ratio
                iweight =  iweight/nm 
                self.t_weights[i]=iweight
        else:
            for i in range(self.total_patients):
                iweight = 0
                freq = (self.t[i]!=-1).sum().item()
                reverse_ratio = self.total_patients/self.t_freqs[freq]
                #reverse_ratio = self.t_freqs[freq]/self.total_patients
                iweight = reverse_ratio # average weight across all modalities
                self.t_weights[i]=iweight
        # make sure t_weights are standardized
        max_t_weights = self.t_weights.max().item()
        self.t_weights = self.t_weights/max_t_weights
        self.pos_weights = torch.tensor(self.pos_weights)

    def _replace_na(self,tensor):
        return torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    

    def __len__(self):
        return self.y.size(0)

    def __getitem__(self, idx):
        if isinstance(self.x,list):
            output_x = [self._replace_na(ix[idx]).float() for ix in self.x ]
        else:
            output_x = self._replace_na(self.x[idx]).float()

        if isinstance(self.t,list):
            output_t = [it[idx].int() for it in self.t ]
        else:
            output_t= self.t[idx].int()

        if isinstance(self.m,list):
            output_m = [im[idx].int() for im in self.m ]
        else:
            output_m = self.m[idx].int()

        if isinstance(self.y,list):
            output_y = [self._replace_na(iy[idx]) for iy in self.y ]
        else:
            output_y = self._replace_na(self.y[idx])

        if self.c is not None:
            if isinstance(self.c,list):
                output_c = [self._replace_na(ic[idx]) for ic in self.c ]
            else:
                output_c = self._replace_na(self.c[idx])
        else: 
            output_c = None
        
        if self.c_index is not None:
            if isinstance(self.c_index,list):
                output_c_index = [ic[idx] for ic in self.c_index ]
            else:
                output_c_index = self.c_index[idx]
        else: 
            output_c_index = None
        return output_x, output_t,output_m,output_y,output_c,output_c_index,self.t_weights[idx]


def data_split(data,which_y,ratio=0.8):
    '''
    Functions to split the tensor dataset according to their labels with considering the number of visits.  
    Input:
        Tensor dataset (outcomes)
    Output:
        Indexes for corresponding splits.
    '''
    # initialize the lists and dictionaries
    splits_n= {} 
    splits_ind={} # nvisits: {label: [index1, index2]}
    nvisits_labels_n = {}
    nvisits_labels_ind = {}

    labels=np.unique(data.y[which_y])
    npatients=data.total_patients

    # calculate the patients with certain number of visits
    for ip in range(npatients):
        # grab information of ip patient
        _,it,_,iy,*_= data[ip]
        label=iy[which_y].item()
            # get number of visits
        if isinstance(it,list):
            it =  torch.maximum(*it) 
        it = it.numpy()
        nvisits= (it!=-1).sum()
        
        if nvisits not in nvisits_labels_n.keys():
           nvisits_labels_n[nvisits] = {}
           nvisits_labels_n[nvisits][label] = 1
           nvisits_labels_ind[nvisits]={}
           nvisits_labels_ind[nvisits][label]=[ip]
        elif label not in nvisits_labels_n[nvisits].keys():
           nvisits_labels_n[nvisits][label] = 1
           nvisits_labels_ind[nvisits][label] = [ip]
        else:
           nvisits_labels_n[nvisits][label] += 1
           nvisits_labels_ind[nvisits][label].append(ip)

    # initialize split dictinoary
    for invisits in nvisits_labels_n.keys():
        if invisits not in splits_ind.keys():
            splits_ind[invisits] = {}
        for label in nvisits_labels_n[invisits].keys():
            if label not in splits_ind[invisits].keys():
                splits_ind[invisits][label] = []

    for invisits in nvisits_labels_n.keys():
        for label in nvisits_labels_n[invisits].keys():
            inum =nvisits_labels_n[invisits][label]
            n1=math.floor(inum*ratio)
            n2=inum - n1     
            ind_perm = np.random.permutation(inum) ## randomness comes from here       
            splits_ind[invisits][label].append([nvisits_labels_ind[invisits][label][i] for i in ind_perm[:n1]])
            splits_ind[invisits][label].append([nvisits_labels_ind[invisits][label][i] for i in ind_perm[n1:]])

    return splits_ind




