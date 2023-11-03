from tqdm import tqdm, trange
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter 
from Model import *
from Dataset import *
from Utils import *

# Global Parameter Definition
CLIENT_PER_MODEL = 5    # numbers of client for each model, total client will be 4* this const
MODEL_TYPE = [Resnet18,Resnet50,Mobilenet_v3,ShuffleNet]    
KL_ROUND = 10           # numbers of updating model with KL div loss
KD_ROUND = 100          # numbers of updating models with Knowledge Distillation method
GLOBAL_EPOCH = 20       # epochs for pretraing by global global dataset
EPOCH = 20              # epochs for pretraing by global local dataset
TRAIN_BATCH_SIZE = 64   # batch size
INFER_BATCH_SIZE = 10000
LR = 0.01             # learning rate
DOWNLOAD = False
IF_USE_GPU = True
DATA_PATH = r'C:\Users\nckubot65904\Desktop\5\code\paper_implement\FedDyn-master\Data\CIFAR10_25_iid_'
SAVE_PATH = r'C:\Users\nckubot65904\Desktop\5\code\paper_implement\model_save'
PROJECT = 'KtpFL_2class_C20_pr02_test'
# PROJECT = 'test'
num_worker = 4
penalty_ratio = 0.2
public_rate = 0.5       # seperate test dataset into public dataset and test dataset, this control the size of public dataset


if __name__ == '__main__':
    torch.backends.cudnn.benchmark=True
    model = Resnet18(num_classes=10)
    optim = torch.optim.SGD(model.parameters(), lr=LR)
    model.cuda()
    private_dataset, public_dataset, test_dataset = Noniid_dataset_5class(10)
    # print(set(private_dataset[0]["train_Y"]))
    # print(set(private_dataset[0]["valid_Y"]))    
    root_path = r'C:\Users\nckubot65904\Desktop\5\code\paper_implement\cifar10'

    trainset = torchvision.datasets.CIFAR10(root=root_path,
                                            train=True,
                                            download=False,
                                            transform=None)
    
    testset = torchvision.datasets.CIFAR10(root=root_path,
                                           train=False,
                                           download=False,
                                           transform=None)
    Train_loaders = (Data.DataLoader(dataset=BaseDataset(datasets=trainset.data,
                                                         labels=trainset.targets,
                                                         transform='train'),
                                     batch_size=TRAIN_BATCH_SIZE,
                                     shuffle=True))
    Val_loaders = (Data.DataLoader(dataset=BaseDataset(datasets=testset.data,
                                                       labels=testset.targets,
                                                       transform='valid'),
                                   batch_size=len(private_dataset[0]["valid_X"]),
                                   shuffle=True))

    # Train_loaders = (Data.DataLoader(dataset=BaseDataset(datasets=private_dataset[0]["train_X"],
    #                                                      labels=private_dataset[0]["train_Y"],
    #                                                      transform='valid'),
    #                                  batch_size=TRAIN_BATCH_SIZE,
    #                                  shuffle=True))
    # Val_loaders = (Data.DataLoader(dataset=BaseDataset(datasets=private_dataset[0]["valid_X"],
    #                                                    labels=private_dataset[0]["valid_Y"],
    #                                                    transform='valid'),
    #                                batch_size=len(private_dataset[0]["valid_X"]),
    #                                shuffle=True))
    # loss = nn.KLDivLoss(reduction='mean')
    loss = nn.CrossEntropyLoss()
    for epoch in range(300):
        model.train()
        total_loss = 0
        with tqdm(total=len(Train_loaders)) as pbar:
            pbar.set_description('Training')
            for iter, (x, y) in enumerate(Train_loaders):
                x = x.cuda()
                y = y.cuda()
                output = model(x)
                losses = loss(log_softmax(output,dim=-1,T=2.0), y)
                # losses = loss(output, y)
                total_loss += losses.data
                optim.zero_grad()
                losses.backward()
                optim.step()
                train_loss = total_loss/(iter+1)
                pbar.update(1)

        print(f'Epoch {epoch:>3}/300, train loss: {train_loss}',end='\t')
        val_loss = 0.0
        val_acc = 0
        model.eval()
        with torch.no_grad():
            for x, y in Val_loaders:
                x = x.cuda()
                y = y.cuda()
                output = model(x)
                losses = loss(log_softmax(output,dim=-1,T=2.0), y)
                # losses = loss(output, y)
                _, predicted = torch.max(output.data, 1)
                # _, y_idx = torch.max(y, 1)

                val_acc += (predicted == y).sum().item()
                val_loss += losses.data
        val_loss = val_loss/len(Val_loaders)
        val_acc = val_acc/len(Val_loaders.dataset.datasets)
        print(f'valid loss: {val_loss}  valid acc: {val_acc}',end='\n')