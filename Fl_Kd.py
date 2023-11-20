import os
import shutil
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
MODEL_TYPE = [ShuffleNet,Mobilenet_v3,Resnet18,Resnet50]
PRIVATE_ROUND = 5       # numbers of updating model with private datasets
KL_ROUND = 3            # numbers of updating model with KL div loss
KD_ROUND = 400          # numbers of updating models with Knowledge Distillation method
GLOBAL_EPOCH = 20       # epochs for pretraing by global global dataset
EPOCH = 20              # epochs for pretraing by global local dataset
TRAIN_BATCH_SIZE = 64   # batch size
INFER_BATCH_SIZE = 10000
LR = 0.001             # learning rate 0.0001
DOWNLOAD = False
IF_USE_GPU = True
SAVE_PATH = r'C:\Users\nckubot65904\Desktop\5\code\paper_implement\model_save'
method = 'KtpFl'
classes = 5
PROJECT = f'{method}_{classes}classuniV_C{CLIENT_PER_MODEL*len(MODEL_TYPE)}_R{KD_ROUND}_KLwopre_sgd'
# PROJECT = 'test'
num_worker = 10
penalty_ratio = 0.2
public_rate = 0.5       # seperate test dataset into public dataset and test dataset, this control the size of public dataset
# private_dataset = Noniid_dataset_2class(20)
# loader = Data.DataLoader(dataset=BaseDataset(datasets=private_dataset[0]["train_X"],labels=private_dataset[0]["train_Y"]),batch_size=10)
# for x,y in loader:
#     print(y)
#     break

if __name__ == '__main__':
### Try to implement FedMD
###   Init models, optimizers and dataloaders
    torch.backends.cudnn.benchmark=True
    Models = []
    for m in MODEL_TYPE:
        for i in range(CLIENT_PER_MODEL):
            Models.append(m(num_classes=10).cuda())
    data_distribution = locals()[f'Noniid_dataset_{classes}class_uniformvalid']

    private_dataset, public_dataset, test_dataset = data_distribution(len(Models))
    
    Optimizers = []
    Train_loaders = []
    Val_loaders = []
    Model_Identify = []
    for idx, model in enumerate(Models):
        Optimizers.append(torch.optim.SGD(model.parameters(), lr=LR))

        if idx==0:
            print(private_dataset[idx]["train_X"].shape)
            print(private_dataset[idx]["train_Y"].shape)
            print(private_dataset[idx]["valid_X"].shape)
            print(private_dataset[idx]["valid_Y"].shape)
            print(public_dataset["X"].shape)
            print(public_dataset["Y"].shape)
            print(test_dataset["X"].shape)
            print(test_dataset["Y"].shape)
        Train_loaders.append(Data.DataLoader(dataset=BaseDataset(datasets=private_dataset[idx]["train_X"],
                                                                 labels=private_dataset[idx]["train_Y"],
                                                                 transform='valid'),
                                             batch_size=TRAIN_BATCH_SIZE,
                                             shuffle=True))
        Val_loaders.append(Data.DataLoader(dataset=BaseDataset(datasets=private_dataset[idx]["valid_X"],
                                                               labels=private_dataset[idx]["valid_Y"],
                                                               transform='valid'),
                                           batch_size=len(private_dataset[idx]["valid_X"]),
                                           shuffle=True))
        Model_Identify.append(f'{idx} {private_dataset[idx]["class"]}')
    Global_dataset = BaseDataset(datasets=public_dataset["X"],
                                 labels=public_dataset["Y"],
                                 transform='valid')
    Global_loader = Data.DataLoader(dataset=Global_dataset, batch_size=TRAIN_BATCH_SIZE)
    INFER_BATCH_SIZE = len(Global_dataset)
    Test_dataset = BaseDataset(datasets=test_dataset["X"],
                               labels=test_dataset["Y"],
                               transform='valid')
    Test_loader = Data.DataLoader(dataset=Test_dataset, batch_size=len(Test_dataset))

    # Loss
    CE_Loss = "CrossEntropy"
    KL_Loss = 'KL_Div'

    pool = mp.Pool(processes=num_worker)
    writer = None
    if os.path.exists(os.path.join(SAVE_PATH, PROJECT)):
        shutil.rmtree(os.path.join(SAVE_PATH, PROJECT))
    os.mkdir(os.path.join(SAVE_PATH, PROJECT))
    os.mkdir(os.path.join(SAVE_PATH, PROJECT, 'init'))
    os.mkdir(os.path.join(SAVE_PATH, PROJECT, 'pretrain'))
    os.mkdir(os.path.join(SAVE_PATH, PROJECT, PROJECT))
    os.mkdir(os.path.join(SAVE_PATH, PROJECT, 'transfer'))
    if PROJECT != 'test':
        writer = SummaryWriter(os.path.join(SAVE_PATH, PROJECT, 'history'))
    
    Load_Path = []
    for m_idx, (model, optim) in enumerate(zip(Models, Optimizers)):
        path = os.path.join(SAVE_PATH, PROJECT, '{}', str(m_idx))
        torch.save({'model':model.state_dict(),'optim':optim.state_dict()}, path.format('init'))
        Load_Path.append(path)
    
###   Pre-Train by global dataset and then by local dataset

    pretrain_task = []
    for m_idx, (model, optim, path, trn_loader, val_loader) in enumerate(zip(Models, Optimizers, Load_Path, Train_loaders, Val_loaders)):
        pretrain_task.append([model,
                              optim,
                              path.format('init'),
                              150,
                              CE_Loss,
                              Global_loader, 
                              val_loader,
                              f'Model {m_idx:3} Global',
                              path.format('pretrain'),
                              Model_Identify[m_idx],
                              ])
    # pretrained = pool.starmap(train, pretrain_task)

    # for midx, history in enumerate(pretrained):
    #     for key, values in history.items():
    #         for iter, value in enumerate(values):
    #             if writer:
    #                 writer.add_scalar(f'pre/{midx}/{key}', value, iter)

###     Test model before KD method

    accBeforekd = []
    for m_idx, (model,path) in enumerate(zip(Models, Load_Path)):

        model_optim = torch.load(path.format('init'), map_location=device)
        model.load_state_dict(model_optim['model'])
        accuracy = 0
        loss = 0
        with torch.no_grad():
            model.eval()
            model.cuda()
            for (x,y) in Test_loader:
                x = x.cuda()
                y = y.cuda()
                output = model(x)
                losses = nn.CrossEntropyLoss()(output, y)
                loss += losses.data
                _, predicted = torch.max(output.data, 1)
                _, y_idx = torch.max(y, 1)
                accuracy += (predicted == y_idx).sum().item()
        print(f'Model: {m_idx:2}  acc: {(accuracy/len(Test_dataset)):.5f}  '
            f'loss: {(loss/len(Test_loader)):.5f}')
        accBeforekd.append(accuracy/len(Test_dataset))

#######################################
###   Train models with KD methods  ###
#######################################

    print(f'------ Start Federated Knowledge Distillation Training ------\n')
    kd_base_dataset = BaseDataset(datasets=public_dataset["X"],
                                  labels=public_dataset["Y"],
                                  transform='default')
    kd_loader = Data.DataLoader(dataset=kd_base_dataset,
                                batch_size=INFER_BATCH_SIZE,
                                shuffle=False)
    coefficient_matrix = torch.ones(len(Models), len(Models), requires_grad=True)
    coefficient_matrix = coefficient_matrix / len(Models)
    for kd_round in range(KD_ROUND):
        #   Train by private data
        private_task = []
        if kd_round%1 == 0:
            for m_idx, (model, optim, path, trn_loader, val_loader) in enumerate(zip(Models, Optimizers, Load_Path, Train_loaders, Val_loaders)):
                if kd_round == 0:
                    model_dir = 'init'
                else:
                    model_dir = PROJECT
                private_task.append([model,
                                    optim,
                                    path.format(model_dir),
                                    PRIVATE_ROUND,
                                    CE_Loss,
                                    trn_loader,
                                    val_loader,
                                    f'private training {m_idx:2}',
                                    path.format(PROJECT),
                                    Model_Identify[m_idx],
                                    ])
            
            private_trained = pool.starmap(train, private_task)
        #   get soft prediction

        softpred_task = []
        for model, path in zip(Models, Load_Path):
            if kd_round == 0:
                model_dir = 'pretrain'
            else:
                model_dir = PROJECT
            softpred_task.append([model, path.format(PROJECT), kd_loader])
        outputs = pool.starmap(get_soft_output, softpred_task)

        #   combine soft prediction
        if method != 'KtpFl':
            soft_y = get_soft_prediction(outputs)
            kd_base_dataset.labels = soft_y
        else:
            models_logits, coefficient_matrix = get_models_logits(outputs, coefficient_matrix, len(Models), penalty_ratio)
            with open(os.path.join(SAVE_PATH, 'coefficient_matrix.txt'),'a') as f:
                f.write(f'{kd_round}\n{str(coefficient_matrix)}\n')
            kd_base_dataset.labels = models_logits.detach().numpy()[0]
        kd_loader = Data.DataLoader(dataset=kd_base_dataset,
                                    batch_size=TRAIN_BATCH_SIZE,
                                    shuffle=False)
        
        #   Train with KD method
        KL_task = []
        print(f"start train {kd_round} rounds kd")
        for m_idx, (model, optim, path, val_loader) in enumerate(zip(Models, Optimizers, Load_Path, Val_loaders)):
            if kd_round == 0:
                model_dir = 'pretrain'
            else:
                model_dir = PROJECT
            KL_task.append([model,
                            optim,
                            path.format(PROJECT),
                            KL_ROUND,
                            KL_Loss,
                            kd_loader,
                            val_loader,
                            f'Model {m_idx:2} Round: {kd_round:3}/{KD_ROUND}',
                            path.format(PROJECT),
                            Model_Identify[m_idx],
                            ])
        
        KL_trained = pool.starmap(train, KL_task)
        for midx, history in enumerate(KL_trained):
            for key, value in history.items():
                if writer:
                    writer.add_scalar(f'kl/{midx}/{key}', value[-1], kd_round)

###################################################
###   Train models with local transfer methods  ###
###################################################
    print(f'------ Start Local Transfer Learning Training ------\n')
    KL_task = []
    for m_idx, (model, optim, path, trn_loader, val_loader) in enumerate(zip(Models, Optimizers, Load_Path, Train_loaders, Val_loaders)):
        KL_task.append([model,
                        optim,
                        path.format('pretrain'),
                        50,
                        CE_Loss,
                        trn_loader,
                        val_loader,
                        f'Model {m_idx:3} Round: {kd_round:2}/{KD_ROUND}',
                        path.format('transfer'),
                        ])
    
    # local = pool.starmap(train, KL_task)

    # for midx, history in enumerate(local):
    #     for key, values in history.items():
    #         for iter, value in enumerate(values):
    #             if writer:
    #                 writer.add_scalar(f'local/{midx}/{key}', value, iter)

###   Test models after KD methods
    accAfterkd = []
    for m_idx, (model,path) in enumerate(zip(Models, Load_Path)):
        model_optim = torch.load(path.format(PROJECT), map_location=device)
        model.load_state_dict(model_optim['model'])
        accuracy = 0
        loss = 0
        model.eval()
        model.cuda()
        with torch.no_grad():
            for (x,y) in Test_loader:
                x = x.cuda()
                y = y.cuda()
                output = model(x)
                losses = nn.CrossEntropyLoss()(output, y)
                loss += losses.data
                _, predicted = torch.max(output.data, 1)
                _, y_idx = torch.max(y, 1)
                accuracy += (predicted == y_idx).sum().item()
        print(f'Model: {m_idx:2}  Test acc: {(accuracy/len(Test_dataset)):.5f}  '
            f'Test loss: {(loss/len(Test_loader)):.5f}')
        accAfterkd.append(accuracy/len(Test_dataset))
        # model.cpu()

###   Test models after local transfer methods
    # accAftertransfer = []
    # for m_idx, (model,path) in enumerate(zip(Models, Load_Path)):
    #     model_optim = torch.load(path.format('transfer'), map_location=device)
    #     model.load_state_dict(model_optim['model'])
    #     accuracy = 0
    #     loss = 0
    #     model.eval()
    #     model.cuda()
    #     with torch.no_grad():
    #         for (x,y) in Test_loader:
    #             x = x.cuda()
    #             y = y.cuda()
    #             output = model(x)
    #             losses = nn.CrossEntropyLoss()(output, y)
    #             loss += losses.data
    #             _, predicted = torch.max(output.data, 1)
    #             _, y_idx = torch.max(y, 1)
    #             accuracy += (predicted == y_idx).sum().item()
    #     print(f'Model: {m_idx:2}  Test acc: {(accuracy/len(Test_dataset)):.5f}  '
    #         f'Test loss: {(loss/len(Test_loader)):.5f}')
    #     accAftertransfer.append(accuracy/len(Test_dataset))
        # model.cpu()

    # for idx, (acc1, acc2, acc3) in enumerate(zip(accBeforekd, accAfterkd, accAftertransfer)):
    #     print(f'Model {idx:2}: {acc3:.4f} -> acc {acc1:.4f} -> {acc2:.4f}')
    
    for idx, (acc1, acc2) in enumerate(zip(accBeforekd, accAfterkd)):
        print(f'Model {idx:2} {Models[idx].get_name():15}: acc {acc1:.4f} -> {acc2:.4f}')
        
    for midx in range(len(Models)):
        if writer:
            writer.add_scalar(f'Test/{midx}', accBeforekd[midx], 0)
            writer.add_scalar(f'Test/{midx}', accAfterkd[midx], 1)

