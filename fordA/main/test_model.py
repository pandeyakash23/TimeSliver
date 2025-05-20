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
from sklearn.metrics import balanced_accuracy_score, confusion_matrix,accuracy_score,roc_auc_score, precision_score, recall_score
from timesliver import dataset, timesliver_network

## Dataloader
batch_size = 256
  

def make_dataset(): 
        
    ohe_valid = np.load('../data/x_test.npy', allow_pickle=True)
    classes_valid = np.argmax(ohe_valid, axis=2)
    output_valid = np.load('../data/y_test.npy', allow_pickle=True)
    seq_len_valid = np.array([ohe_valid.shape[1]]*len(ohe_valid))
    sax_valid = np.load('../data//sax_test.npy', allow_pickle=True)
  
 
    test_dataset = dataset(ohe_valid,sax_valid,\
        classes_valid,seq_len_valid,output_valid,ohe_valid.shape[0])

      
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False)   
    
    return  test_loader, ohe_valid.shape[0]

    
def initalize():
    global rank
    rank = 'cuda:5'
    model_dict  =np.load('./model/save_dict.npy', allow_pickle=True).tolist()
    model = timesliver_network(2,\
        model_dict['q'],model_dict['d'],model_dict['max_m'],rank)
    model.load_state_dict(torch.load('./model/best.pth'))
    # model = torch.load('./model/best.pth')
    # print(model)
    # rank = next(model.parameters()).device 
    model.eval().to(rank) 
    print('Number of trainable parameters:', builtins.sum(p.numel() for p in model.parameters()))
    criterion = nn.CrossEntropyLoss()
    
    return model, criterion

def test():
    test_loader, valid_size  = make_dataset()
    model, criterion = initalize()
    rank = next(model.parameters()).device 
    with torch.no_grad():
        predicted_label = torch.zeros((valid_size, 1))
        predicted_prob_class1 = torch.zeros((valid_size, 1))
        actual_label = torch.zeros((valid_size, 1))
        count_valid = 0         
        for j, (i_x,sax, i_classes, i_seq, i_actual) in enumerate(test_loader):
            i_x = i_x.to(rank) #.type(dtype=torch.float32)
            sax = sax.to(rank) #.type(dtype=torch.float32)
            i_seq = i_seq.to(rank).type(dtype=torch.float32)
            i_classes = i_classes.to(rank)
            i_actual = i_actual.to(rank)
            
           # forward pass     
            iter_y_pred = model(i_x, sax, i_seq)
            size = iter_y_pred.size(0)
            # loss = criterion(iter_y_pred, i_actual)
            # valid_loss = (valid_loss*j + loss.item())/(j+1)
            iter_y_pred = nn.Softmax(dim=1)(iter_y_pred)
            predicted_prob_class1[count_valid:count_valid+size, 0] = iter_y_pred[:,1]
            iter_y_pred = torch.argmax(iter_y_pred, dim=1)
            predicted_label[count_valid:count_valid+size, 0] = iter_y_pred 
            actual_label[count_valid:count_valid+size, 0] = i_actual
            count_valid += size
        
        predicted_label = predicted_label.cpu().numpy().reshape((-1,1))
        predicted_prob_class1 = predicted_prob_class1.cpu().numpy().reshape((-1,1))
        actual_label = actual_label.cpu().numpy().reshape((-1,1))
        valid_acc = accuracy_score(actual_label, predicted_label)
        print(f'Test accuracy is: {valid_acc}')
        print(confusion_matrix(actual_label, predicted_label))

        
if __name__=='__main__':
    cp_1 = time.time()
    test()
    cp_2 = time.time()
    print('Time Taken',cp_2-cp_1)
