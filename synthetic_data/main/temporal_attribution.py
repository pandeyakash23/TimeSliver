import torch
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn import preprocessing
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
from sklearn.metrics import balanced_accuracy_score, confusion_matrix,mean_absolute_error,r2_score, mean_squared_error
from timesliver import dataset, timesliver_network


which_data = input('Enter the dataset for which you want to calculate the token importance (train,valid,test):')

## Dataloader
batch_size = 1024
# class spiderdataset(Dataset) :
#     def __init__(self,ohe, classes,seq_len,output, n_samples) :
#         # data loading
#         self.ohe = torch.from_numpy(ohe.astype(np.float32))
#         self.seq_len = torch.from_numpy(seq_len.astype(int64))
#         self.classes = torch.from_numpy(classes.astype(int64)) 
#         self.output = torch.from_numpy(output.astype(np.int64)).reshape((n_samples,))
#         self.n_samples = n_samples 
        
#     def __getitem__(self,index) :
#         return self.ohe[index], self.classes[index], self.seq_len[index], self.output[index]

#     def __len__(self):    
#         return self.n_samples      

def make_dataset(): 
        
    ohe_valid = np.load(f'../data/x_{which_data}.npy', allow_pickle=True)
    classes_valid = np.argmax(ohe_valid, axis=2)
    output_valid = np.load(f'../data/y_{which_data}.npy', allow_pickle=True)
    seq_len_valid = np.array([ohe_valid.shape[1]]*len(ohe_valid))   
    sax_valid = np.load(f'../data//sax_{which_data}.npy', allow_pickle=True)
 
    test_dataset = dataset(ohe_valid,sax_valid,
                                 classes_valid,seq_len_valid,output_valid,ohe_valid.shape[0])

      
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False)   
    
    return  test_loader, ohe_valid.shape[0], ohe_valid.shape[1]

    
def initalize():    
    init_lr = np.load('./model/init_lr.npy', allow_pickle=True)
    global rank
    rank = 'cuda:4'
    model_dict  =np.load('./model/save_dict.npy', allow_pickle=True).tolist()
    model = timesliver_network(2,\
        model_dict['q'],model_dict['d'],model_dict['max_m'],rank)
    model.load_state_dict(torch.load('./model/best.pth'))
    # model = torch.load('./model/best.pth')
    # print(model)
    # rank = next(model.parameters()).device 
    model.eval().to(rank) 
    print('Number of trainable parameters:', builtins.sum(p.numel() for p in model.parameters()))
    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    
    return model, criterion,optimizer

def motif_identification():  
    
    test_loader, test_size, max_seq_len  = make_dataset()
    model, criterion,optimizer = initalize()
    rank = next(model.parameters()).device 
    store_importance = torch.zeros((test_size, max_seq_len)).to(rank)
    count_test = 0
    for _, (i_x,sax,i_classes, i_seq, i_actual) in enumerate(test_loader):
        i_x = i_x.to(rank) #.type(dtype=torch.float32)
        sax = sax.to(rank) #.type(dtype=torch.float32)
        i_seq = i_seq.to(rank).type(dtype=torch.float32)
        # i_classes = i_classes.to(rank)
        i_actual = i_actual.to(rank)
        i_batch = len(i_actual)
        iter_y_pred, cam,initial_cam,initial_pool = \
            model.forward_motif_importance(i_x, sax, i_seq)
        base_loss = criterion(iter_y_pred, i_actual)
        
        # temp_idx = i_actual.to('cpu').tolist()
        # iter_y_pred = nn.Softmax(dim=-1)(iter_y_pred)[:,temp_idx] ## taking class prob
        # # iter_y_pred = torch.argmax(iter_y_pred, dim=1)
        # i_actual = torch.zeros(iter_y_pred.size()).to(rank)
        # base_loss = criterion(iter_y_pred, i_actual)
        
        optimizer.zero_grad()
        base_loss.backward()
        
        # cam = torch.abs(cam[0])
        # cam = nn.ReLU()(cam[0])
        cam = cam[0]
        print('Size of the gradient',cam.size())
        
        ## initial_cam
        
        # initial_cam = torch.abs(initial_cam[0])
        initial_cam = initial_cam[0]
        print('Size of the initial_cam',initial_cam.size())
        # print(aaa)
        
        # ## initial_cam
        # initial_pool = torch.abs(initial_pool[0])
        # print('Size of the initial_pool',initial_pool.size())
        # # print(aaa)

        
        for prot in range(cam.size(0)):
            cam[prot,...] = cam[prot,...]/abs(torch.max(cam[prot,...])+1E-18)
            initial_cam[prot,...] = \
                initial_cam[prot,...]/abs(torch.max(initial_cam[prot,...])+1E-18)
        #     # initial_pool[prot,...] = \
        #     #     initial_pool[prot,...]/(torch.max(initial_pool[prot,...])+1E-18)
        
            
        for m_i in range(cam.size(-1)):
            mo_level_imp = \
                model.calculate_motif_level_new(cam[...,m_i], m_i+1, initial_cam)
            

            kernel_size = max_seq_len - mo_level_imp.size(-1) + 1
            store_importance[count_test:count_test+i_batch,...] += model.assigning_importance(mo_level_imp, \
                kernel_size, max_seq_len)

            head_kernel = kernel_size/torch.arange(1,kernel_size)
            head_kernel = head_kernel.to(rank)
            tail_kernel = head_kernel.flip(0)
            
            # print(tail_kernel)
            
            store_importance[count_test:count_test+i_batch,0:kernel_size-1] = \
                store_importance[count_test:count_test+i_batch,0:kernel_size-1]*head_kernel
            
            store_importance[count_test:count_test+i_batch,(max_seq_len-kernel_size+1):max_seq_len] = \
                store_importance[count_test:count_test+i_batch,(max_seq_len-kernel_size+1):max_seq_len]*tail_kernel
            
            
            del mo_level_imp

        
        count_test += i_batch
    
    with torch.no_grad():   
        store_importance = store_importance.to('cpu').numpy()
        print(store_importance[15])
        np.save(f'./model/importance_{which_data}', store_importance)
    

        
if __name__=='__main__':
    cp_1 = time.time() 
    motif_identification()
    cp_2 = time.time()
    print('Time Taken',cp_2-cp_1)
    
    
