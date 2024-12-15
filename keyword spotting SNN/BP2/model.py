#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import time
import tqdm
import random

import pandas as pd

# global parameters
thresh = 0.6  # neuronal threshold
lens = thresh  # hyper-parameters of approximate function
decay = 0.2    # decay constants
MAX_TIME = 130 # max time intervel


# Relu-like activation function
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
 
    def forward(self, x):
        x = x * torch.sigmoid(10*x)
        return x


# fire-control with proper backward defined
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        scope = abs(input - thresh) <= lens
        is_spike = input >= thresh
        return grad_input * (1.0 + is_spike.float()) * scope.float()
act_fun_default = ActFun.apply


# update method of membrane-potential among time
def mem_update(fc_input, x, mem, spike, act_fun):
    mem_t = mem * decay * (1. - spike)  + fc_input(x)
    spike = act_fun(mem_t) # act_fun : approximation firing function
    return mem_t, spike


# learning-rate scheduler
def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5
    return optimizer


# define LIF neuron model / verision that across time-dimension
class LIF_neurons_vias_time(nn.Module):
    def __init__(self, n_neurons, max_time=MAX_TIME, act_fun=act_fun_default, device=torch.device("cpu")):
        super(LIF_neurons_vias_time, self).__init__()
        
        self.n_neurons = n_neurons
        self.device = device
        self.max_time = max_time

        self.h_mem = None
        self.h_spike = None
        
        self.h_spike_l = None
        
        self.act_fun = act_fun
        
    def forward(self,input):
        batch_size = input.shape[0]
        init_mem(batch_size)
        
        for t in range(self.max_time):
            x = input[:,t]
            self.h_mem, self.h_spike = self.mem_update(x, self.h_mem, self.h_spike, self.act_fun)
            self.h_spike_l[:,t] = self.h_spike

        return self.h_spike_l
    
    def init_mem(self, batch_size):
        self.h_mem = torch.zeros(batch_size, self.n_neurons, device=self.device)
        self.h_spike = torch.zeros(batch_size, self.n_neurons, device=self.device)
        self.h_spike_l = torch.zeros(batch_size, self.max_time, self.n_neurons, device=self.device)
    
    def mem_update(self, input, mem, spike, act_fun):
        mem_t = mem * decay * (1. - spike)  + input
        spike = self.act_fun(mem_t) # act_fun : approximation firing function
        return mem_t, spike



# define LIF neuron model / verision that deal with one transfer
class LIF_neurons(nn.Module):
    def __init__(self, n_neurons, act_fun=act_fun_default, device=torch.device("cpu")):
        super(LIF_neurons, self).__init__()
        
        self.n_neurons = n_neurons
        self.device = device

        self.h_mem = None
        self.h_spike = None
        
        self.act_fun = act_fun
        
    def forward(self,input):
        self.h_mem, self.h_spike = self.mem_update(input, self.h_mem, self.h_spike, self.act_fun)
        return self.h_spike
    
    def init_mem(self, batch_size):
        self.h_mem = torch.zeros(batch_size, self.n_neurons, device=self.device)
        self.h_spike = torch.zeros(batch_size, self.n_neurons, device=self.device)
    
    def mem_update(self, input, mem, spike, act_fun):
        mem_t = mem * decay * (1. - spike)  + input
        spike = self.act_fun(mem_t) 
        return mem_t, spike



def init_weights(m):
    if type(m) == nn.Linear:
        print("init")
        torch.nn.init.xavier_normal_(m.weight)



#---------------------------------
#=================================

import matplotlib.pyplot as plt
import numpy as np
def print_results(acc_record, loss_test_record):
    plt.plot(range(len(acc_record)), acc_record)
    plt.xlabel('epoch')
    plt.ylabel('acc[%]')
    plt.show()
    plt.plot(range(len(loss_test_record)), loss_test_record)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()



def test_snn(snn, args, test_loader, device=None):
    correct = 0
    total = 0
    loss_total = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
            for step, (test_data,test_label) in enumerate(test_loader):
                test_label = test_label.to(device)
                test_data = test_data.float().to(device)

                outputs = snn(test_data)

                _, predicted = outputs.max(1)

                batch_size = test_data.shape[0]
                labels_ = torch.zeros(batch_size, args.n_classes).scatter_(1, test_label.view(-1, 1), 1)
                loss = criterion(outputs.cpu(), labels_)
                loss_total += loss.item()
                
                total += float(test_label.shape[0])
                correct += float(predicted.eq(test_label.squeeze(-1)).sum().item())

            acc = 100. * float(correct) / float(total)
            
            return acc, loss_total




def train_snn(snn, args, train_loader, test_loader, device=None):

    acc_record = list([])
#     loss_train_record = list([])
    loss_test_record = list([])

    snn = snn.apply(init_weights)
    criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(snn.parameters(), lr=args.lr)
    for epoch in range(args.num_epochs):
        running_loss = 0
        start_time = time.time()
        total = 0
        correct = 0
        bar = tqdm.tqdm(total = len(train_loader))
        bar.update(0)


        for step, (input_data,labels) in enumerate(train_loader):

            optimizer.zero_grad()
            batch_size = input_data.shape[0]

            input_data = input_data.float().to(device)
            labels = labels.to(device)

            # outputs = snn(input_data,act_fun)
            outputs = snn(input_data)

            _, predicted = outputs.max(1)
            #print(predicted.shape)
            #print(labels.shape)
            total += float(labels.shape[0])
            #print(predicted.eq(labels))
            correct += float(predicted.eq(labels.squeeze(-1)).sum().item())
            # print(total,correct)

            labels_ = torch.zeros(batch_size, args.n_classes).to(device).scatter_(1, labels.view(-1, 1), 1)
            loss = criterion(outputs, labels_)
            # loss.requires_grad = True
    #         loss = criterion(outputs.cpu(), labels.reshape(-1))
            running_loss += loss.item()
            loss.backward()

            optimizer.step()
            bar.update(1)
            if (step+1)%50 == 0:
                print ('\nEpoch [%d/%d], Step [%d/%d], Loss: %.5f'
                        %(epoch+1, args.num_epochs, step+1, len(train_loader),running_loss ))
#                 loss_train_record.append(running_loss)
                running_loss = 0
                train_acc = correct/total
                print('Accuracy:', train_acc)
                print('Time elasped:', time.time()-start_time)
                correct = 0
                total = 0
        listtrain = [epoch,train_acc]
        data = pd.DataFrame([listtrain])
        data.to_csv('E:\\gam001_train_acc.csv',mode='a',header=False,index=False)#mode设为a,就可以向csv文件追加数据了

        optimizer = lr_scheduler(optimizer, epoch, args.lr, 10)
        
        print('Iters:', epoch)
        test_acc, test_loss = test_snn(snn, args, test_loader, device=device)
        acc_record.append(test_acc)
        loss_test_record.append(test_loss)
        print('Test Accuracy on test dataset: %.3f' % (test_acc))  
        print('Test Loss on test dataset: %.3f' % (test_loss))  
        print('Time elasped:', time.time()-start_time)
        print(f"Maximum accuracy on test dataset: {max(acc_record)}")
        print('\n\n\n')

        step = "Step[%d]"%epoch
        t_loss = "%f"%test_loss
        t_acc = "%g"%test_acc
        listt = [epoch,t_loss,t_acc]
        data = pd.DataFrame([listt])
        data.to_csv('E:\\gam001_test_acc.csv',mode='a',header=False,index=False)#mode设为a,就可以向csv文件追加数据了

    return acc_record, loss_test_record



class SNN_sample(nn.Module):
    def __init__(self, device='cpu', n_input=620, n_class=11):
        super(SNN_sample, self).__init__()
        self.device = device
    
        # define layers of LIF neurons
        self.h1 = LIF_neurons(120) # hidden layer with 120 LIF neurons
        self.h2 = LIF_neurons(11)  # hidden layer with 11 LIF neurons


        # network setup
        self.net = nn.Sequential(
            nn.Linear(n_input, 120, bias = False),
            Swish(),
            self.h1,
            nn.Linear(120, n_class, bias = False),
            Swish(),
            self.h2,
        )
        
    def forward(self, input):

        batch_size = input.shape[0]
        time_window = input.shape[1]
        sumspike = torch.zeros(batch_size, 11, device=self.device) # init sumspike

        # initialize membrane-potential of LIF neurons
        self.h1.init_mem(batch_size)
        self.h2.init_mem(batch_size)

        # simulation time steps
        for t in range(time_window): 
            x = input[:,t]
            out_spike = self.net(x)
            sumspike += out_spike

        # take sum of spikes as prediction
        outputs = sumspike / 50
        return outputs


class SNN_sample_2(nn.Module):
    def __init__(self, device='cpu', n_input=620, n_class=11):
        super(SNN_sample_2, self).__init__()
        self.device = device
    
        # define layers of LIF neurons
        self.h0 = LIF_neurons(200) # hidden layer with 120 LIF neurons
        self.h1 = LIF_neurons(120) # hidden layer with 120 LIF neurons
        self.h2 = LIF_neurons(11)  # hidden layer with 11 LIF neurons


        # network setup
        self.net = nn.Sequential(
            nn.Linear(n_input, 200, bias = False),
            Swish(),
            self.h0,
            nn.Linear(200, 120, bias = False),
            Swish(),
            self.h1,
            nn.Linear(120, n_class, bias = False),
            Swish(),
            self.h2,
        )
        
    def forward(self, input):

        batch_size = input.shape[0]
        time_window = input.shape[1]
        sumspike = torch.zeros(batch_size, 11, device=self.device) # init sumspike

        # initialize membrane-potential of LIF neurons
        self.h0.init_mem(batch_size)
        self.h1.init_mem(batch_size)
        self.h2.init_mem(batch_size)

        # simulation time steps
        for t in range(time_window): 
            x = input[:,t]
            out_spike = self.net(x)
            sumspike += out_spike

        # take sum of spikes as prediction
        outputs = sumspike / 50
        return outputs