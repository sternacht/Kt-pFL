import torch
import os
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import multiprocessing as mp
from Model import *
from Utils import device, temperature_scaled_softmax

def mdtrain(model, optim, path, epoches, loss_func, trn_loader, val_loader=None, outstr=''):
    model_optim = torch.load(path, map_location=device)
    model.load_state_dict(model_optim['model'])
    optim.load_state_dict(model_optim['optim'])
    model.cuda()
    out = ''
    patience = 5
    for epoch in range(epoches):
        total_loss = 0.0
        model.train()
        for x, y in trn_loader:
            x = x.cuda()
            y = y.cuda()
            output = model(x)
            # output = temperature_scaled_softmax(output, dim=1, T=1.0)
            loss = loss_func(output, y)
            total_loss += loss.data
            optim.zero_grad()
            loss.backward()
            optim.step()
        out += f'{outstr}  Epoch:{epoch+1:3}  Train Loss: {(total_loss/len(trn_loader)):.5f}'
        if val_loader != None:
            min_loss = np.inf
            val_loss = 0.0
            max_acc = 0.0
            acc = 0
            model.eval()
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.cuda()
                    y = y.type(torch.LongTensor)
                    y = y.cuda()
                    output = model(x)
                    output = temperature_scaled_softmax(output, dim=1, T=1.0)
                    loss = loss_func(output, y)
                    _, predicted = torch.max(output.data, 1)
                    acc += (predicted == y).sum().item()
                    val_loss += loss.data
                val_loss = val_loss/len(val_loader)
                acc = acc/len(val_loader.dataset)
                out += f'  Valid Loss: {val_loss:.5f} Acc: {acc:.4f}\n'
                if acc > max_acc:
                    patience = 5
                    max_acc = acc
                    torch.save({'model':model.state_dict(),'optim':optim.state_dict()}, path)
                else:
                    patience -= 1
                    if patience == 0:
                        break
        else:
            torch.save({'model':model.state_dict(),'optim':optim.state_dict()}, path)
    print(out)
    return 

if __name__ == "__main__":
    EPOCH = 30                #全部data訓練10次
    BATCH_SIZE = 64           #每次訓練隨機丟50張圖像進去
    LR =0.001                 #learning rate
    LR_decay = 0.9
    DOWNLOAD_MNIST = False    #第一次用要先下載data,所以是True
    if_use_gpu = 1            #使用gpu
    SAVE_PATH = r'C:\Users\nckubot65904\Desktop\5\code\paper_implement\model_save'
    PROJECT = 'FedMD_pretrain_test'
    num_worker = 4
    model_structure = [{'c':[128,256],       'drop':0.2},
                       {'c':[128,128,192],   'drop':0.2},
                       {'c':[64 ,64 ,64],    'drop':0.2},
                       {'c':[128,64 ,64],    'drop':0.3},
                       {'c':[64 ,64 ,128],   'drop':0.4},
                       {'c':[64 ,128,256],   'drop':0.2},
                       {'c':[64 ,128,192],   'drop':0.2},
                       {'c':[128,192,256],   'drop':0.2},
                       {'c':[128,128,128],   'drop':0.3},
                       {'c':[64 ,64 ,64 ,64],'drop':0.2},
                      ]
    # coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
    #                            3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
    #                            6, 11,  5, 10,  7,  6, 13, 15,  3, 15, 
    #                            0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
    #                            5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
    #                            16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
    #                            10, 3,  2, 12, 12, 16, 12,  1,  9, 19, 
    #                            2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
    #                            16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
    #                            18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    # train_data.targets = coarse_labels[train_data.targets]
    train_data = torchvision.datasets.CIFAR10(
        root='./cifar10/',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=DOWNLOAD_MNIST
    )

    train_loader = Data.DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle=True)

    test_data = torchvision.datasets.CIFAR10(
        root='./cifar10/', 
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=DOWNLOAD_MNIST,
        )
    # test_data.targets = coarse_labels[test_data.targets]
    test_loader = Data.DataLoader(dataset = test_data, batch_size = BATCH_SIZE, shuffle=True)
    # test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1).float(), requires_grad=False)
    if not os.path.exists(os.path.join(SAVE_PATH, PROJECT)):
        os.mkdir(os.path.join(SAVE_PATH, PROJECT))
    # test_y = test_data.test_labels
    Model_Optim_Path = []
    for idx, s in enumerate(model_structure):
        model = Costum_nlayer_CNN(c=s['c'],num_classes=10,drop_rate=s['drop']).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        path = os.path.join(SAVE_PATH, PROJECT, str(idx))
        torch.save({'model':model.state_dict(),'optim':optimizer.state_dict()}, path)
        Model_Optim_Path.append([model,optimizer,path])


    loss_function = nn.CrossEntropyLoss()
    pool = mp.Pool(processes=num_worker)

    pretrain_task = []
    for m_idx, mop in enumerate(Model_Optim_Path):
        model, optim, path = mop
        # mdtrain(model,
        #         optim,
        #         path,
        #         EPOCH,
        #         loss_function, 
        #         train_loader,
        #         test_loader,
        #         f'Model {m_idx+1:3}')
        pretrain_task.append([model,
                              optim,
                              path,
                              EPOCH,
                              loss_function, 
                              train_loader,
                              test_loader,
                              f'Model {m_idx+1:3}'])
        
    pretrained = pool.starmap(mdtrain, pretrain_task)