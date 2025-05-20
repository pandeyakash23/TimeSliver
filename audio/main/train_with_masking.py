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
from sklearn.metrics import balanced_accuracy_score, confusion_matrix,accuracy_score
from timesliver import dataset, timesliver_network

parser = argparse.ArgumentParser(description='TimeSliver_masking')
parser.add_argument('--num_epochs', default=2500, type=int,
                    metavar='N',
                    )
parser.add_argument('--device', default='cuda', type=str
                    )

top_per = input('Top how much percentage tokens should remain unmasked? ')
top_per = int(top_per)
np.save('./model/top_per', top_per)

which_imp = 'main'
np.save('./model/which_imp', which_imp)

writer = SummaryWriter(f"Training starting on:{date.today()}")
if which_imp=='transformer':
    sub_method = input('Sub method? attn or grad?')
    np.save('./model/sub_method', sub_method)
    writer = SummaryWriter(comment=f"TimeSliver_masking, imp:{which_imp}_{sub_method}, percentage:{top_per}")

elif which_imp=='captum':
    sub_method = input('Sub method? ig/gs/dl/dlshap?')
    np.save('./model/sub_method', sub_method)
    writer = SummaryWriter(comment=f"TimeSliver_masking, imp:{which_imp}_{sub_method}, percentage:{top_per}")

else:
    writer = SummaryWriter(comment=f"TimeSliver_masking, imp:{which_imp}, percentage:{top_per}")


## Dataloader
batch_size = 512

args = parser.parse_args()
num_epochs = args.num_epochs
device = args.device

print('Device', args.device)


def masking_function(ohe, seq_len, importance):
    revised_x  = ohe
    fea_size = ohe.shape[-1]
    num_ex = ohe.shape[0]
    for k in range(num_ex):
        l = int(seq_len[k])
        ex_token = np.argsort(importance[k,0:l], axis=-1)
        ex_token = ex_token[::-1]
        top_num_token = int(ceil(l*top_per/100))
        sample_imp = tuple(ex_token[top_num_token:].tolist())
        corrupt = np.random.normal(0,1,size=(len(sample_imp),fea_size)).tolist()
        # revised_x[k,sample_imp,:] = corrupt
        revised_x[k,sample_imp,:] = 0
    return revised_x  

def load_individual_data(path, name):  
    ohe = np.load(f'{path}x_{name}.npy', allow_pickle=True)
    if which_imp not in ['transformer', 'captum']:
        imp_token = np.load(f'../{which_imp}/model/importance_{name}.npy', allow_pickle=True)
    else: 
        imp_token = np.load(f'../{which_imp}/model/importance_{name}_{sub_method}.npy', allow_pickle=True)
    sax_train = np.load(f'../data//sax_{name}.npy', allow_pickle=True)
    
    classes = np.argmax(ohe, axis=2)
    output = np.load(f'{path}/y_{name}.npy', allow_pickle=True)
    seq_len = np.array([ohe.shape[1]]*len(ohe))
    if top_per < 100:
        ohe = masking_function(ohe, seq_len, imp_token) 
        sax_train = masking_function(sax_train, seq_len, imp_token) 

    print(ohe.shape)
    return ohe,sax_train,classes,seq_len,output
    

def make_dataset():        
    path = '../data/'
    
    ohe,sax_train,classes,seq_len,output = load_individual_data(path, 'train')
    ohe_valid,sax_valid,classes_valid,seq_len_valid,output_valid = load_individual_data(path, 'valid')
    
    global q
    q = ohe.shape[-1]
    
    train_dataset = dataset(ohe,sax_train,classes,seq_len,output,ohe.shape[0])      
    valid_dataset = dataset(ohe_valid,sax_valid,classes_valid,seq_len_valid,output_valid,ohe_valid.shape[0])      
    
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True)  
      
    valid_loader = DataLoader(dataset=valid_dataset,
                            batch_size=batch_size,
                            shuffle=False) 
    
    
    return train_loader, valid_loader, len(ohe_valid)
    
def initalize(rank, max_m, init_lr):
    q = 40
    d = 12
    num_classes = 10 ## one property prediction
    model = timesliver_network(num_classes, q,d,max_m,rank).to(rank)     
    print('Number of trainable parameters:', builtins.sum(p.numel() for p in model.parameters()))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    
    ## saving q,d,max_m for later use
    save_dict = {'q':q, 'd':d, 'max_m':max_m}
    np.save('./model/save_dict.npy', save_dict) 
    
    return model, criterion, optimizer

def eval_loop(model,valid_loader, valid_size,criterion,rank):
    
    model.eval()
    
    with torch.no_grad():
        predicted_label = torch.zeros((valid_size, 1))
        actual_label = torch.zeros((valid_size, 1))
        count_valid = 0       
        # valid_loss = 0  
        for j, (i_x, sax, i_classes, i_seq, i_actual) in enumerate(valid_loader):
            i_x = i_x.to(rank) #.type(dtype=torch.float32)
            sax = sax.to(rank) #.type(dtype=torch.float32)
            i_seq = i_seq.to(rank).type(dtype=torch.float32)
            i_classes = i_classes.to(rank)
            i_actual = i_actual.to(rank)
            
            # forward pass     
            iter_y_pred = model(i_x, sax, i_seq)
            # loss = criterion(iter_y_pred, i_actual)
            # valid_loss = (valid_loss*j + loss.item())/(j+1)
            iter_y_pred = nn.Softmax(dim=1)(iter_y_pred)
            iter_y_pred = torch.argmax(iter_y_pred, dim=1)
            size = iter_y_pred.size(0)
            predicted_label[count_valid:count_valid+size, 0] = iter_y_pred 
            actual_label[count_valid:count_valid+size, 0] = i_actual
            count_valid += size
        
        predicted_label = predicted_label.cpu().numpy().reshape((-1,1))
        actual_label = actual_label.cpu().numpy().reshape((-1,1))
        valid_acc = accuracy_score(actual_label, predicted_label)
    
    model.train()    
    return valid_acc
    
## Training loop
def train(num_epochs, init_lr, max_m):
    rank = device
    train_loader, valid_loader, valid_size = make_dataset()
    model, criterion, optimizer = initalize(rank, max_m, init_lr)
    start_from = 0
    largest_acc = 0
    
    best_valid = 0
    
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

        valid_acc = eval_loop(model,valid_loader, valid_size,criterion,rank)
        
        if valid_acc>best_valid:
            best_valid = valid_acc 
            
                    
        writer.add_scalar("Cross entropy Loss per epoch/train", avg_loss, epoch+1+start_from)
        writer.add_scalar("Valid Acc", valid_acc, epoch+1+start_from)
        
        if valid_acc >= largest_acc:
            torch.save(model, f'./model/best_masking.pth')
            largest_acc = valid_acc
            
        
if __name__=='__main__':
    cp_1 = time.time()
    init_lr = 0.001
    np.save('./model/init_lr', init_lr)
    max_m = int(1)
    ##change
    train(num_epochs, init_lr, max_m)
    cp_2 = time.time()
    print('Time Taken',cp_2-cp_1)
