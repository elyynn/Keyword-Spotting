import argparse
import data #a build in
import model #is written in model.py
from model import train_snn, print_results, SNN_sample, SNN_sample_2

import torch #building neural networks
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

import torchvision #handling vision-related tasks
import torchvision.transforms as transforms
import os

import pandas as pd #

dftrain = pd.DataFrame(columns=['epoch','training accuracy'])#列名
dftrain.to_csv("train_acc.csv",index=False) #路径可以根据需要更改
dftest = pd.DataFrame(columns=['epoch','train Loss','test accuracy'])#列名
dftest.to_csv("test_acc.csv",index=False) #路径可以根据需要更改
#save logging to csv


parser = argparse.ArgumentParser()


parser.add_argument('--mat_file', type=str, help='Path to a MAT file')
parser.add_argument("--dataset",type=str,default="TIDIGITS")
parser.add_argument("--train_batch_size",type=int,default=44)    # 训练集上单次并行的样本数

# batch: divide the train and test data into several groups to use
# Why: Loading the entire dataset into memory might not be feasible
# helps in obtaining a more stable estimate of the gradient

parser.add_argument("--eval_batch_size",type=int,default=22)     # 测试集上单次并行的样本数
parser.add_argument("--lr",type=float,default=3e-3)             # learning rate
# learning rate


parser.add_argument("--num_epochs",type=int,default=50)      # 训练周期 默认一直训练
parser.add_argument("--gpu_id",type=int,default=-1)
# The gpu_id allows you to specify which GPU to use if you have multiple GPUs available in your system. 
# If you only have one GPU, the gpu_id is typically 0. If you want to use the CPU instead, 
# you can set gpu_id to -1.             
parser.add_argument("--criterion",type=str,default="MSE")       
# The “criterion” in the context of machine learning and neural networks refers 
# to the loss function or cost function used to evaluate the performance of the model. 
# Mean Squared Error
parser.add_argument("--n_classes",type=int,default=11)          # 类别数量
parser.add_argument("--n_input",type=int,default=620)           # 输入维度 （频率）
parser.add_argument("--max_time",type=int,default=100)          # 时间维度
# Time Dimension:
#this usually refers to the length of the time sequence or 
# the number of frames/steps in each input sample.
# max

args = parser.parse_args()
args.dataset_path = "./combined_data.mat" # 数据集路径
# so if not using add_argument for dataset_path, each time the code have to be changed to change path


if args.gpu_id == -1:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:{}".format(args.gpu_id))
# Set the device to CPU or GPU
#  in PyTorch, you can specify which device (CPU or GPU) to use for tensor computations using torch.device

train_set, test_set = data.load_data(args)
# train_set = TensorDataset(train_datasets,train_labels)
train_loader = DataLoader(train_set,batch_size=args.train_batch_size,shuffle=True,drop_last=False)
# test_set = TensorDataset(test_datasets,test_labels)
test_loader = DataLoader(test_set,batch_size=args.eval_batch_size,shuffle=False,drop_last=False)
# DataLoader is a utility that helps in efficiently loading and iterating over 
# datasets during the training or evaluation of models


snn = SNN_sample(device=device, n_input=args.n_input, n_class=args.n_classes)
snn = snn.to(device)

acc_record, loss_test_record = train_snn(snn, args, train_loader, test_loader, device=device)

print_results(acc_record, loss_test_record)

