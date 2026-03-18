import torch
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import r2_score
import random
import matplotlib as mpl
import os
import gc
import pandas as pd
import csv
from numpy import *
from torch.utils.tensorboard import SummaryWriter
from datetime import date
import time
import builtins
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


# amino_acid = np.load('./model/categorical_variables.npy', allow_pickle=True)
# amino_acid = amino_acid.tolist()

   
class dataset(Dataset) :
    def __init__(self,ohe, sax, classes,seq_len,output, n_samples) :
        # data loading
        self.ohe = torch.from_numpy(ohe.astype(np.float32))
        self.sax = torch.from_numpy(sax.astype(np.float32))
        self.seq_len = torch.from_numpy(seq_len.astype(int64))
        self.classes = torch.from_numpy(classes.astype(int64)) 
        self.output = torch.from_numpy(output.astype(np.int64)).reshape((n_samples,))
        self.n_samples = n_samples
        
        
    def __getitem__(self,index) :
        return self.ohe[index], self.sax[index],self.classes[index], self.seq_len[index], self.output[index]

    def __len__(self):    
        return self.n_samples      

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.05, max_len: int = 18000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        if d_model%2 != 0:
            div_term = torch.exp(torch.arange(0, d_model+1, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, 1, d_model+1)
        else:
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, 1, d_model)
            
        position = torch.arange(max_len).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        if d_model%2!=0:
            pe = pe[:,:,0:-1]
        self.register_buffer('pe', pe)

    def forward(self, x, rank):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        # print(x.size(), self.pe[:x.size(0)].size())
        x = x + self.pe[:x.size(0)].to(rank)
        return self.dropout(x)

def initial_q(grad):
        # grad = torch.mean(grad, dim=1)
        # return grad
        initial_out.append(grad)

def initial_d(grad):
        # grad = torch.mean(grad, dim=1)
        # return grad
        initial_pool.append(grad)

def extract(grad):
        # grad = torch.mean(grad, dim=1)
        # return grad
        cam.append(grad)
          
class complor_network(nn.Module):
    def __init__(self, num_classes, d_model, d_out, max_m, rank):
        super(complor_network, self).__init__()
        self.max_m = max_m
        self.rank = rank
        self.d_in = d_model
        self.d_model = d_model ## equivalent to q
        self.d_out = d_out ## equivalent to d
        sax_alpha = 40
        # self.d

        ##n layers
        self.proj = nn.Linear(self.d_in, self.d_model)
        # cnn layers   
        self.positional_encoding = PositionalEncoding(self.d_model)
        self.cnn1 = nn.Sequential( nn.Conv1d(self.d_model,16,2, stride=1), 
                                   nn.ReLU(),
                                   nn.Conv1d(16,32,2, stride=1),
                                   nn.ReLU(),
                                   nn.Conv1d(32,int(self.d_out),2, stride=1)
                                   ) ##8 size motifs
        self.ln1 = nn.LayerNorm(self.d_out)

        self.ln4 = nn.LayerNorm(sax_alpha)
        
        self.maxpool1 = nn.Sequential( nn.AvgPool1d(2, stride=1),
                                       nn.AvgPool1d(2, stride=1),
                                       nn.AvgPool1d(2, stride=1),
                                   )

        
        self.motif_size_1 = 4*1

        
        self.reduction = nn.Sequential(nn.AvgPool2d((2,2), stride=(2,2)),
                                    #    nn.AvgPool2d((2,2), stride=(2,2)),
                                    #    nn.AvgPool2d((2,1), stride=(2,1))
                                       )
        
        self.nn = nn.Sequential(
                                # nn.Linear(int(sax_alpha*self.d_out*self.max_m),num_classes),
                                nn.Linear(int(360*self.max_m),num_classes),
                                # nn.ReLU(),
                                # nn.Linear(32,8),
                                # nn.ReLU(),
                                # nn.Linear(8,num_classes)
                                )
        
    def make_p(self, out, sax, seq_len,m):
        cnn_net = getattr(self, f'cnn{m}')
        pool_net = getattr(self, f'maxpool{m}')
        ln_net = getattr(self, f'ln{m}')
        mo_size = getattr(self, f'motif_size_{m}')
        
        q =  cnn_net(out) ##[N,f,L]  
        q = torch.permute(ln_net(torch.permute(q,(0,2,1))),(0,2,1))
        
        store_q = q
       
        ## changing pooling
        d = pool_net(sax)*mo_size ## [N,f,L]
        
        # for i in range(out.size(0)):
        d = d/seq_len[0]  
        
        q = torch.matmul(d, q.permute(0,2,1))
        q = q.reshape((q.size(0), q.size(1), q.size(2),1)) 
        
        
        return store_q, d, q
        
        
    def forward(self, x, sax, seq_len):
        'x: [batch, seq_len, feature], classes: [N,L]'
        
        out = self.proj(x)
        
        # out = self.positional_encoding(out.permute(1,0,2),self.rank) ##[L,N,f]        
        # out = out.permute(1,2,0)
        
        out = torch.permute(out,(0,2,1)) ## making it (N, f, L)
        sax = sax.permute(0,2,1)
        
        _,_,out_1 = self.make_p(out, sax,seq_len,1)
        # _,_,out_2 = self.make_p(out, sax,seq_len,2)
        
        # heat_map = torch.cat((out_1, out_2), dim=3)      
        heat_map = out_1
        # heat_map = nn.Softmax(dim=1)(heat_map)
        
        heat_map = heat_map.permute(0,2,3,1)
        
        # heat_map = self.ln4(heat_map)
        # print('Before', heat_map.size())
        heat_map = heat_map.squeeze()
        heat_map = self.reduction(heat_map)
        # print('after', heat_map.size())
        
        
        heat_map = nn.Flatten()(heat_map) ## removing amino acid contribution from heat map as there are none        
        heat_map = self.nn(heat_map)
        
        return heat_map
    
            
    def assigning_importance(self,mo_level, kernel_size,unwrapped_len):   
        reduced_len = mo_level.size(-1)
        sequence_importance = torch.zeros((mo_level.size(0), unwrapped_len)).to(self.rank)
        
        for i in range(reduced_len):
            sequence_importance[:,i:(i+kernel_size)] += \
                mo_level[:,i].unsqueeze(-1)
        
        return sequence_importance
        
    def sum_subsequent_n(self,temp, n):
        if n > len(temp):
            raise ValueError("n should be less than or equal to the length of temp")
        
        windowed_view = np.lib.stride_tricks.sliding_window_view(temp, n)
        
        return windowed_view.sum(axis=1)   
    
    @torch.no_grad()
    def calculate_motif_level(self,dp,m_i, initial_cam,P):
        d_comp = getattr(self, f'pool_{m_i}') #self.pool_1[:,q,:]
        q_comp = getattr(self, f'out_{m_i}')
        m_size = initial_cam.size(1)-(d_comp.size(-1)-1)
    
        total = d_comp.size(0)
        max_len = d_comp.size(-1)
        q_id = q_comp.size(1)
        d_id = d_comp.size(1)
        
        all_motif_importance = torch.zeros((total, max_len)).to(self.rank)
        temp = torch.zeros((total,d_comp.size(-1))).to(self.rank)
        max_val_prot = torch.zeros((total,1)).to(self.rank)
        var_imp = torch.zeros((total,1)).to(self.rank)
        signs = torch.zeros((total,1)).to(self.rank)
        for i in range(q_id): # i is related to size of latent rep
            for j in range(d_id): # j in related to num of categorical var
                # for prot in range(total): # prot defines the protein number in a batch
                    # var_imp[:,:] = (P[:,j,i]*dp[:,j,i]).unsqueeze(-1)
                    var_imp[:,:] = (dp[:,j,i]).unsqueeze(-1)
                    signs[:,:] = torch.sign(var_imp)
                    
                    
                    temp[:,:] = nn.ReLU()(signs*d_comp[:,j,:]*q_comp[:,i,:])
                    # temp[:,:] = nn.Sigmoid()(signs*d_comp[:,j,:]*q_comp[:,i,:])
                    # temp[:,:] =d_comp[:,j,:]*q_comp[:,i,:]
                    # temp[:,:] = torch.abs(d_comp[:,j,:]*q_comp[:,i,:])
                    # temp[:,:] = nn.ReLU()(signs*d_comp[:,j,:]*q_comp[:,i,:]) + nn.ReLU()(-signs*d_comp[:,j,:]*q_comp[:,i,:])
                    # temp[:,:] = torch.abs(signs*d_comp[:,j,:]*q_comp[:,i,:]*initial_cam[:,i,:])
                    # temp[:,:] = nn.ReLU()(signs*d_comp[:,j,:]*q_comp[:,i,:]*initial_cam[:,i,:])
                    # temp_sum = torch.sum(temp, dim=1).unsqueeze(-1)
                    # temp = temp/(temp_sum+1E-18)

                    # max_val_prot[:,0] = torch.max(abs(temp), dim=1)[0].squeeze()
                    # temp[:,:] = (temp)/(max_val_prot+1E-18)                  
                    all_motif_importance += temp*abs(var_imp)              
                    # all_motif_importance += temp*var_imp 


        return all_motif_importance
    
    def forward_motif_importance(self, x,  sax, seq_len):
        'x: [batch, seq_len, feature], classes: [N,L]'
        global cam
        global initial_out, initial_pool
        cam = []
        initial_out = []
        initial_pool = []
        self.Lin = x.size(1)
        self.seq_len = seq_len
        # self.trace_visitation = trace_visitation
        # self.importance =  importance
        # self.overall_imp_segments = overall_imp_segments

        out = self.proj(x)
        
        # out = self.positional_encoding(out.permute(1,0,2),self.rank) ##[L,N,f]        
        # out = out.permute(1,2,0)
        
        out = torch.permute(out,(0,2,1)) ## making it (N, f, L)
        sax = sax.permute(0,2,1)
        
        self.out_1, self.pool_1,p_1 = self.make_p(out, sax,seq_len,1)
        self.out_1.register_hook(initial_q)
        # self.pool_1.register_hook(initial_d)
        
        # self.out_2, self.pool_2,p_2 = self.make_p(out, sax,seq_len,2)
        
        # heat_map = torch.cat((p_1, p_2), dim=3)    
        heat_map = p_1
        heat_map.register_hook(extract)
    
        heat_map = heat_map.permute(0,2,3,1)  
        
        # heat_map = self.ln4(heat_map) 
        
        heat_map = heat_map.squeeze()
        heat_map = self.reduction(heat_map)
          
        heat_map = nn.Flatten()(heat_map) ## removing amino acid contribution from heat map as there are none        
        heat_map = self.nn(heat_map)

        
        return heat_map, cam,initial_out,p_1