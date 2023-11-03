import numpy as np
import copy
import torch
import torch.utils.data as Data
import torch.nn as nn
# from torch.nn.functional import log_softmax
from sklearn.model_selection import train_test_split
from Dataset import BaseDataset

device = torch.device("cuda:0")

def temperature_scaled_softmax(logits, dim=-1, T=1.0):
    logits = logits / T
    return torch.softmax(logits, dim=dim)

def softmax(logits):
    return torch.softmax(logits, dim=-1)

def log_softmax(logits, dim=-1, T=1.0):
    logits = logits / T
    return torch.nn.functional.log_softmax(logits, dim=dim)

def get_soft_prediction(outputs):
    if len(outputs) == 0:
        return None
    elif len(outputs) ==1:
        return outputs[0]
    else:
        soft_y = outputs[0].copy()
        for output in outputs[1:]:
            soft_y += output
        soft_y = soft_y / len(outputs)
    return soft_y

def split_val_dataset(val_data, val_label, num_clients, bs=64):
    Val_loaders = []
    vox, voy = val_data, val_label
    for i in range(num_clients, 1, -1):
        #   'o': other
        #   'i': ith client
        vox, vix, voy, viy = train_test_split(vox,
                                              voy,
                                              test_size=1/i,
                                              stratify=voy)
        Val_loader = Data.DataLoader(dataset=BaseDataset(datasets=vix,
                                                         labels=viy),
                                     batch_size=bs,
                                     shuffle=False)
        Val_loaders.append(Val_loader)
    Val_loader = Data.DataLoader(dataset=BaseDataset(datasets=vox,
                                                     labels=voy),
                                 batch_size=bs,
                                 shuffle=False)
    Val_loaders.append(Val_loader)
    return Val_loaders

def train(model, optim, path, epoches, loss_func, trn_loader, val_loader=None, outstr='',save_path=None, identify=''):
    if save_path == None:
        save_path = path
    if loss_func == 'KL_Div':
        loss = nn.KLDivLoss(reduction='batchmean')
    else:
        loss = nn.CrossEntropyLoss()
    model_optim = torch.load(path, map_location=device)
    model.load_state_dict(model_optim['model'])
    optim.load_state_dict(model_optim['optim'])
    # model.cuda()
    model.zero_grad()
    out = ''
    patience = 5
    max_acc = 0.0
    history = {}
    history['Train Loss'] = []
    if val_loader != None:
        history['Valid Loss'] = []
        history['Valid Acc'] = []
    for epoch in range(epoches):
        total_loss = 0.0
        model.train()
        predict = np.zeros(10).astype(np.int64)
        groundtruth = np.zeros(10).astype(np.int64)
        for iter, (x, y) in enumerate(trn_loader):
            x = x.cuda()
            y = y.cuda()
            output = model(x)
            if epoch == 0 and iter == 0:
                print(identify, softmax(output[0]),y[0])
            if loss_func == 'KL_Div':
                losses = loss(log_softmax(output,dim=-1,T=2.0), y)
            else:
                # output = temperature_scaled_softmax(output, dim=-1, T=1.0)
                losses = loss(output, y)
            _,a = torch.max(output.data, 1)
            for i in a.cpu().detach().numpy():
                predict[i] += 1
            _,b = torch.max(y, 1)
            for j in b.cpu().detach().numpy():
                groundtruth[j] += 1
            total_loss += losses.data
            optim.zero_grad()
            losses.backward()
            optim.step()
        if epoch == 0:
            ret = ''
            for p,g in zip(predict,groundtruth):
                ret += f'{p}/{g}  '
            print(identify, ret)
        out += f'{outstr}   Epoch:{epoch:3}  Train Loss: {(total_loss/len(trn_loader)):.5f}'
        history['Train Loss'].append(total_loss/len(trn_loader))
        torch.save({'model':model.state_dict(),'optim':optim.state_dict()}, save_path)
        if val_loader != None:
            min_loss = np.inf
            val_loss = 0.0
            acc = 0
            model.eval()
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.cuda()
                    y = y.cuda()
                    output = model(x)
                    if loss_func == 'KL_Div':
                        losses = loss(log_softmax(output,dim=-1,T=2.0), y)
                    else:
                        # output = temperature_scaled_softmax(output, dim=-1, T=1.0)
                        losses = loss(output, y)
                    _, predicted = torch.max(output.data, 1)
                    _, y_idx = torch.max(y, 1)
                    acc += (predicted == y_idx).sum().item()
                    val_loss += losses.data
            val_loss = val_loss/len(val_loader)
            acc = acc/len(val_loader.dataset.datasets)
            out += f'  Valid Loss: {val_loss:.5f} Acc: {acc:.4f}\n'
            history['Valid Loss'].append(val_loss)
            history['Valid Acc'].append(acc)
            if acc > max_acc:
                patience = 5
                max_acc = acc
                torch.save({'model':model.state_dict(),'optim':optim.state_dict()}, save_path)
            else:
                # patience -= 1
                if patience == 0:
                    break
        else:
            torch.save({'model':model.state_dict(),'optim':optim.state_dict()}, save_path)
    print(out)
    return history

def pretrain(model, optim, path, epoches, loss_func, data_loader, val_loader=None, outstr=['',''], save_path=None):
    if save_path == None:
        save_path = path
    his = train(model, 
          optim,
          path,
          epoches[0],
          loss_func[0], 
          data_loader[0], 
          val_loader=val_loader,
          outstr=outstr[0],
          save_path=save_path)
    his = train(model, 
          optim, 
          save_path,
          epoches[1], 
          loss_func[1], 
          data_loader[1], 
          val_loader=val_loader,
          outstr=outstr[1],
          save_path=save_path)
    return his

def get_soft_output(model, path, kd_loader, T=2, identify=''):
    model_optim = torch.load(path, map_location=device)
    model.load_state_dict(model_optim['model'])
    client_output = None
    predicted = np.zeros(10).astype(np.int32)
    groundtruth = np.zeros(10).astype(np.int32)
    with torch.no_grad():
        model.eval()
        model.cuda()
        for iter, (x, y) in enumerate(kd_loader):
            x = x.cuda()
            y = y.cuda()
            output = model(x)
            output = temperature_scaled_softmax(output, dim=-1, T=T)
            _, predict = torch.max(output.data, 1)
            _, y_idx = torch.max(y, 1)
            _,a = torch.max(output.data, 1)
            _,b = torch.max(y, 1)
            for i, j in zip(a.cpu().detach().numpy(), b.cpu().detach().numpy()):
                if (i==j):
                    predicted[i] += 1
                groundtruth[j] += 1
            ret = ''
            for p,g in zip(predicted,groundtruth):
                ret += f'{p}/{g}  '
            # if iter == 0:
            #     print(f'{identify} soft {(predict==y_idx).sum().item()}/{len(output)}')
            if isinstance(client_output, np.ndarray):
                client_output = np.concatenate([client_output, output.cpu().numpy()])
            else:
                client_output = output.cpu().numpy()
        model.cpu()
        # print(identify, ret)
    return client_output
    
def test(model, path, loss, loader, output):
    model_optim = torch.load(path.format('pretrain'), map_location=device)
    model.load_state_dict(model_optim['model'])
    accuracy = 0
    loss = 0
    with torch.no_grad():
        model.eval()
        model.cuda()
        for x, y in loader:
            x = x.cuda()
            y = y.cuda()
            output = model(x)
            losses = loss(output, y)
            loss += losses.data
            _, predicted = torch.max(output.data, 1)
            _, y_idx = torch.max(y, 1)
            accuracy += (predicted == y_idx).sum().item()
    output += f'acc: {(accuracy/len(loader.datasets)):.5f} loss: {(loss/len(loader.datasets)):.5f}'
    acc = (accuracy/len(loader.datasets))
    return acc, output

def get_models_logits(raw_logits, weight_alpha, N_models, penalty_ratio): #raw_logits为list-np；weight为tensor；
    weight_mean = torch.ones(N_models, N_models, requires_grad=True)
    weight_mean = weight_mean.float()/(N_models)
    loss_fn = torch.nn.KLDivLoss(reduce=True, size_average=True, reduction='batchmean')
    teacher_logits = torch.zeros(N_models, np.size(raw_logits[0],0), np.size(raw_logits[0],1), requires_grad=False) #创建logits of teacher  #next false
    models_logits = torch.zeros(N_models, np.size(raw_logits[0],0), np.size(raw_logits[0],1), requires_grad=True) #创建logits of teacher
    #weight.requires_grad = True #can not change requires_grad here
    weight = weight_alpha.clone()
    for self_idx in range(N_models): #对每个model计算其teacher的logits加权平均值
        teacher_logits_local = teacher_logits[self_idx]
        for teacher_idx in range(N_models): #对某一model，计算其他所有model的logits
            # if self_idx == teacher_idx:
            #     continue
            #teacher_tmp = weight[self_idx][teacher_idx] * raw_logits[teacher_idx]
            #teacher_logits[self_idx] = torch.add(teacher_logits[self_idx], weight[self_idx][teacher_idx] * raw_logits[teacher_idx]) 
            #teacher_logits[self_idx] = torch.add(teacher_logits[self_idx], weight[self_idx][teacher_idx] * torch.autograd.Variable(torch.from_numpy(raw_logits[teacher_idx]))) 
            #teacher_logits[self_idx] = torch.add(teacher_logits[self_idx], weight[self_idx][teacher_idx] * torch.from_numpy(raw_logits[teacher_idx])) 
            teacher_logits_local = torch.add(teacher_logits_local, weight[self_idx][teacher_idx] * torch.from_numpy(raw_logits[teacher_idx])) 
            #                                                                tensor中的一个像素点，本质标量 * teacher的完整logits
            
        loss_input = torch.from_numpy(raw_logits[self_idx])
        #loss_target = torch.autograd.Variable(teacher_logits[self_idx], requires_grad=True)   
        loss_target = teacher_logits_local                    
                                           
        loss = loss_fn(loss_input, loss_target)

        loss_penalty = loss_fn(weight[self_idx], weight_mean[self_idx])
        # print('loss_penalty:', loss_penalty)     
        # print('loss:', loss)
        loss += loss_penalty*penalty_ratio
        #loss = SoftCrossEntropy_without_logsoftmax(loss_input,loss_target)

        #weight[self_idx].zero_grad()
        #weight[self_idx].grad.zero_()
        weight.retain_grad() #保留叶子张量grad
        #print('weight.grad before loss.backward:', weight.grad)
        loss.backward(retain_graph=True)
        # print('weight:', weight)
        #print('weight.requires_grad:', weight.requires_grad)
        #print('weight.grad:', weight.grad)
        #print('weight[self_idx]:', weight[self_idx])
        #print('weight[self_idx].grad:', weight[self_idx].grad)
        with torch.no_grad():
            #weight[self_idx] = weight[self_idx] - weight[self_idx].grad * 0.001  #更新权重
            gradabs = torch.abs(weight.grad)
            gradsum = torch.sum(gradabs)
            gradavg = gradsum.item() / (N_models)
            grad_lr = 1.0
            # for i in range(5): #0.1
            #     if gradavg > 0.01:
            #         gradavg = gradavg*1.0/5
            #         grad_lr = grad_lr/5                
            #     if gradavg < 0.01:
            #         gradavg = gradavg*1.0*5
            #         grad_lr = grad_lr*5
            # print("grad_lr:", grad_lr)
            weight.sub_(weight.grad*grad_lr)
            #weight.sub_(weight.grad*50)
            weight.grad.zero_()
    #############设定权重######################
    # set_weight_local = []
    # weight1 = [0.18, 0.18, 0.18, 0.18, 0.18, 0.02, 0.02, 0.02, 0.02, 0.02]
    # weight2 = [0.02, 0.02, 0.02, 0.02, 0.02, 0.18, 0.18, 0.18, 0.18, 0.18]
    # for i in range(N_models):
    #     if i <= 4:
    #         set_weight_local.append(weight1)
    #     if i >= 5:
    #         set_weight_local.append(weight2)
    # tensor_set_weight_local = torch.Tensor(set_weight_local)
    ###################################
    # 更新 raw_logits
    for self_idx in range(N_models): #对每个model计算其teacher的logits加权平均值
        weight_tmp = torch.zeros(N_models)
        idx_count = 0
        for teacher_idx in range(N_models): #对某一model，计算其softmax后的weight
            # if self_idx == teacher_idx:
            #     continue
            #weight加softmax#
            weight_tmp[idx_count] = weight[self_idx][teacher_idx]
            idx_count += 1
        #softmax_fn = nn.softmax() #这里不对，不应该softmax，应该normalization##先用低温softmax#
        weight_local = nn.functional.softmax(weight_tmp*5.0)

        idx_count = 0
        for teacher_idx in range(N_models): #对某一model，计算其他所有model的logits
            # if self_idx == teacher_idx:
            #     continue
            #models_logits[self_idx] = torch.add(models_logits[self_idx], weight[self_idx][teacher_idx] * torch.from_numpy(raw_logits[teacher_idx]))             
            #设定权重models_logits[self_idx] = torch.add(models_logits[self_idx], tensor_set_weight_local[self_idx][idx_count] * torch.from_numpy(raw_logits[teacher_idx]))
            with torch.no_grad():
                models_logits[self_idx] = torch.add(models_logits[self_idx], weight_local[idx_count] * torch.from_numpy(raw_logits[teacher_idx]))            

                #设定权重weight[self_idx][teacher_idx] = tensor_set_weight_local[self_idx][idx_count]                
                weight[self_idx][teacher_idx] = weight_local[idx_count]
            idx_count += 1             
    # print('weight after softmax:', weight)
    #
    return models_logits, weight