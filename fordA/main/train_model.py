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
import argparse
from sklearn.metrics import balanced_accuracy_score, confusion_matrix,mean_absolute_error, accuracy_score
from timesliver import dataset, timesliver_network


writer = SummaryWriter(f"Training starting on:{date.today()}")
writer = SummaryWriter(comment="TimeSliver model")

parser = argparse.ArgumentParser(description='TimeSliver')
parser.add_argument('--num_epochs', default=5000, type=int,
                    metavar='N',
                    )
parser.add_argument('--device', default='cuda', type=str
                    )
parser.add_argument('--categorical', default=10, type=int
                    )

args = parser.parse_args()
num_epochs = args.num_epochs
device = args.device
categorical = args.categorical

np.save('./model/categorical_size', categorical)
## Dataloader
batch_size = 512

def make_dataset():        
    ohe = np.load('../data/x_train.npy', allow_pickle=True)
    classes = np.argmax(ohe, axis=2)
    output = np.load('../data//y_train.npy', allow_pickle=True)
    seq_len = np.array([ohe.shape[1]]*len(ohe))
    sax_train = np.load(f'../data//sax_train.npy', allow_pickle=True)
    
    global q
    q = ohe.shape[-1]
 
    train_dataset = dataset(ohe,sax_train,classes,seq_len,output,ohe.shape[0])    
        
    ohe_valid = np.load('../data//x_valid.npy', allow_pickle=True)
    classes_valid = np.argmax(ohe_valid, axis=2)
    output_valid = np.load('../data//y_valid.npy', allow_pickle=True)
    seq_len_valid = np.array([ohe_valid.shape[1]]*len(ohe_valid))
    sax_valid = np.load(f'../data//sax_valid.npy', allow_pickle=True)
    
 
    test_dataset = dataset(ohe_valid,sax_valid,classes_valid,seq_len_valid,output_valid,ohe_valid.shape[0])

    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True)  
      
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False)   
    
    return train_loader, test_loader, ohe_valid.shape[0]


    
def initalize(rank, max_m, init_lr):
    d = 36
    num_classes = 2 ## one property prediction
    model = timesliver_network(num_classes, q,d,max_m,rank).to(rank)     
    print('Number of trainable parameters:', builtins.sum(p.numel() for p in model.parameters()))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    
    ## saving q,d,max_m for later use
    save_dict = {'num_classes':num_classes,'q':q, 'd':d, 'max_m':max_m}
    np.save('./model/save_dict.npy', save_dict) 
    
    return model, criterion, optimizer

## Training loop
def train(num_epochs, init_lr, max_m):
    rank = device
    train_loader, valid_loader, valid_size  = make_dataset()
    model, criterion, optimizer = initalize(rank, max_m, init_lr)
    start_from = 0
    largest_acc = 0
    for epoch in range(num_epochs):
        avg_loss = 0
        for i, (i_x,sax,i_classes, i_seq, i_actual) in enumerate(train_loader):
            i_x = i_x.to(rank) #.type(dtype=torch.float32)
            sax = sax.to(rank) #.type(dtype=torch.float32)
            i_seq = i_seq.to(rank).type(dtype=torch.float32)
            i_classes = i_classes.to(rank)
            i_actual = i_actual.to(rank)
            
            # forward pass    
            iter_y_pred = model(i_x, sax, i_seq) ## get the output in [batch, seq_len, feature_size]
            loss = criterion(iter_y_pred, i_actual)
            avg_loss = (avg_loss*i + loss.item())/(i+1)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()   

        with torch.no_grad():
            predicted_label = torch.zeros((valid_size, 1))
            actual_label = torch.zeros((valid_size, 1))
            count_valid = 0       
            valid_loss = 0  
            for j, (i_x, sax, i_classes, i_seq, i_actual) in enumerate(valid_loader):
                i_x = i_x.to(rank) #.type(dtype=torch.float32)
                sax = sax.to(rank) #.type(dtype=torch.float32)
                i_seq = i_seq.to(rank).type(dtype=torch.float32)
                i_classes = i_classes.to(rank)
                i_actual = i_actual.to(rank)
                
                # forward pass     
                iter_y_pred = model(i_x, sax, i_seq)
                loss = criterion(iter_y_pred, i_actual)
                valid_loss = (valid_loss*j + loss.item())/(j+1)
                iter_y_pred = nn.Softmax(dim=1)(iter_y_pred)
                iter_y_pred = torch.argmax(iter_y_pred, dim=1)
                size = iter_y_pred.size(0)
                predicted_label[count_valid:count_valid+size, 0] = iter_y_pred 
                actual_label[count_valid:count_valid+size, 0] = i_actual
                count_valid += size
            
            predicted_label = predicted_label.cpu().numpy().reshape((-1,1))
            actual_label = actual_label.cpu().numpy().reshape((-1,1))
            valid_acc = accuracy_score(actual_label, predicted_label)
            
                    
        writer.add_scalar("Cross entropy Loss per epoch/train", avg_loss, epoch+1+start_from)
        writer.add_scalar("Acc  per epoch/valid", valid_acc, epoch+1+start_from)
        # writer.add_scalar("Cross entropy per epoch/valid", valid_loss, epoch+1+start_from)
        
        if valid_acc >= largest_acc:
            torch.save(model.state_dict(), f'./model/best.pth')
            # torch.save(model, f'./model/best.pth')
            largest_acc = valid_acc
            
        
if __name__=='__main__':
    cp_1 = time.time()
    init_lr = 0.0003
    np.save('./model/init_lr', init_lr)
    max_m = int(1)
    ##change
    train(num_epochs, init_lr, max_m)
    cp_2 = time.time()
    print('Time Taken',cp_2-cp_1)
