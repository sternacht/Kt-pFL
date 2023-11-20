import os
import numpy as np
import torch
import torchvision
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms

Default_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
])

Valid_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
])

CIFAR10_PATH = r'C:\Users\nckubot65904\Desktop\5\code\paper_implement\cifar10'

def get_cifar10():
    trainset = torchvision.datasets.CIFAR10(root=CIFAR10_PATH,
                                            train=True,
                                            download=False,
                                            transform=None)
    testset = torchvision.datasets.CIFAR10(root=CIFAR10_PATH,
                                           train=False,
                                           download=False,
                                           transform=None)
    return trainset, testset

class ImageDataset(Dataset):
    def __init__(self, 
                 idx=0, 
                 transform='dafault', 
                 target_transform=None, 
                 path=None, 
                 train=True):
        PATH = r'C:\Users\nckubot65904\Desktop\5\code\paper_implement\FedDyn-master\Data\CIFAR10_20_Dirichlet_0.600' if path == None else path
        if train:
            if isinstance(idx, int):
                self.datasets = np.load(os.path.join(PATH,'clnt_x.npy'))[idx]
                self.labels = np.load(os.path.join(PATH,'clnt_y.npy'))[idx]
            elif isinstance(idx, list):
                self.datasets = np.load(os.path.join(PATH,'clnt_x.npy'))[idx[0]:idx[-1]+1]
                self.datasets = np.concatenate(self.datasets)
                self.labels = np.load(os.path.join(PATH,'clnt_y.npy'))[idx[0]:idx[-1]+1]
                self.labels = np.concatenate(self.labels)
            else:
                ValueError('training client index should be int or list format')
        else:
            self.datasets = np.load(os.path.join(PATH,'tst_x.npy'))
            self.labels = np.load(os.path.join(PATH,'tst_y.npy'))

        if transform == 'valid':
            self.transform = None
        else:
            self.transform = Default_transforms
        self.target_transform = target_transform
        for idx in range(len(self.datasets)):
            min = np.min(self.datasets[idx])
            max = np.max(self.datasets[idx])
            self.datasets[idx] = (self.datasets[idx] - min)/(max-min)

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        image = self.datasets[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            pass
        if isinstance(label, int):
            one_hot_label = one_hot(torch.from_numpy(label), num_classes=10).float()
        return image, one_hot_label
    
    
class BaseDataset(Dataset):
    def __init__(self, 
                 datasets = None, 
                 labels = None, 
                 transform='default', 
                 target_transform=None):
        self.datasets = datasets
        self.labels = labels
        self.transform = transform
        if transform == 'valid':
            self.transform = Valid_transforms
        else:
            self.transform = Default_transforms
        self.target_transform = target_transform
        # for idx in range(len(self.datasets)):
        #     min = np.min(self.datasets[idx])
        #     max = np.max(self.datasets[idx])
        #     self.datasets[idx] = (self.datasets[idx] - min)/(max-min)

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        image = self.datasets[idx]
        label = self.labels[idx]
        image = self.transform(image).float()
        if self.target_transform:
            pass
        if isinstance(label, np.int32):
            label = torch.from_numpy(np.array([label]).astype(np.int64))
            label = one_hot(label, num_classes=10)[0].float()
        return image, label
    
def Noniid_dataset_2class(N_client, public_rate=0.5):
    trainset, testset = get_cifar10()
    private_dataset = []
    classes = [i for i in range(10)]
    
    for n in range(N_client):
        if len(classes) == 0:
            classes = [i for i in range(10)]
        # client_class = np.random.choice(len(classes),2, replace=False)
        # client_class.sort()
        client_class = classes[:2]
        # private_classes = [classes.pop(client_class[1]),classes.pop(client_class[0])]
        private_classes = [classes.pop(classes.index(client_class[1])),classes.pop(classes.index(client_class[0]))]
        idx = [(c in private_classes) for c in trainset.targets]
        private_x = trainset.data[idx]
        private_y = np.array(trainset.targets)[idx]
        private_trn_x, private_val_x, private_trn_y, private_val_y= train_test_split(private_x, 
                                                                                     private_y, 
                                                                                     test_size=0.25,
                                                                                     stratify=private_y)
        private_dataset.append({"train_X":private_trn_x, "train_Y":private_trn_y,
                                "valid_X":private_val_x, "valid_Y":private_val_y,
                                "class":f'{private_classes[0]},{private_classes[1]}'})
    
    test_x, public_x, test_y, public_y= train_test_split(testset.data, 
                                                         np.array(testset.targets), 
                                                         test_size=public_rate, 
                                                         stratify=np.array(testset.targets))
                                                         
    public_dataset = {"X":public_x,"Y":public_y}
    test_dataset = {"X":test_x,"Y":test_y}

    return private_dataset, public_dataset, test_dataset

def Noniid_dataset_2class_uniformvalid(N_client, public_rate=0.5, valid_size=0.1):
    trainset, testset = get_cifar10()
    X, Y = trainset.data, np.array(trainset.targets)
    train_x, valid_x, train_y, valid_y= train_test_split(X, 
                                                         Y, 
                                                         test_size=valid_size,
                                                         stratify=Y)
    private_dataset = []
    classes = [i for i in range(10)]
    for n in range(N_client):
        if len(classes) == 0:
            classes = [i for i in range(10)]
        client_class = np.random.choice(len(classes),2, replace=False)
        client_class.sort()
        private_classes = [classes.pop(client_class[1]),classes.pop(client_class[0])]
        idx = [(c in private_classes) for c in train_y]
        private_x = train_x[idx]
        private_y = train_y[idx]
        private_dataset.append({"train_X":private_x, "train_Y":private_y,
                                "valid_X":valid_x, "valid_Y":valid_y,
                                "class":f'{private_classes[0]},{private_classes[1]}'})
    test_x, public_x, test_y, public_y= train_test_split(testset.data, 
                                                         np.array(testset.targets), 
                                                         test_size=public_rate, 
                                                         stratify=np.array(testset.targets))
    public_dataset = {"X":public_x,"Y":public_y}
    test_dataset = {"X":test_x,"Y":test_y}

    return private_dataset, public_dataset, test_dataset
    
def Noniid_dataset_3class_uniformvalid(N_client, public_rate=0.5, valid_size=0.1):
    trainset, testset = get_cifar10()
    X, Y = trainset.data, np.array(trainset.targets)
    train_x, valid_x, train_y, valid_y= train_test_split(X, 
                                                         Y, 
                                                         test_size=valid_size,
                                                         stratify=Y)
    private_dataset = []
    classes = [i for i in range(10)]
    for n in range(N_client):
        if len(classes) < 3:
            classes = [i for i in range(10)]
        client_class = np.random.choice(len(classes),3, replace=False)
        client_class.sort()
        private_classes = [classes.pop(client_class[2]),classes.pop(client_class[1]),classes.pop(client_class[0])]
        idx = [(c in private_classes) for c in train_y]
        private_x = train_x[idx]
        private_y = train_y[idx]
        private_dataset.append({"train_X":private_x, "train_Y":private_y,
                                "valid_X":valid_x, "valid_Y":valid_y,
                                "class":f'{private_classes[0]},{private_classes[1]},{private_classes[2]}'})
    test_x, public_x, test_y, public_y= train_test_split(testset.data, 
                                                         np.array(testset.targets), 
                                                         test_size=public_rate, 
                                                         stratify=np.array(testset.targets))
    public_dataset = {"X":public_x,"Y":public_y}
    test_dataset = {"X":test_x,"Y":test_y}

    return private_dataset, public_dataset, test_dataset
        
def Noniid_dataset_5class_uniformvalid(N_client, public_rate=0.5, valid_size=0.1):
    trainset, testset = get_cifar10()
    X, Y = trainset.data, np.array(trainset.targets)
    train_x, valid_x, train_y, valid_y= train_test_split(X, 
                                                         Y, 
                                                         test_size=valid_size,
                                                         stratify=Y)
    private_dataset = []
    classes = [i for i in range(10)]
    for n in range(N_client):
        if len(classes) == 0:
            classes = [i for i in range(10)]
        client_class = np.random.choice(len(classes),5, replace=False)
        client_class.sort()
        private_classes = [classes.pop(client_class[4]),classes.pop(client_class[3]),
                           classes.pop(client_class[2]),classes.pop(client_class[1]),classes.pop(client_class[0])]
        idx = [(c in private_classes) for c in train_y]
        private_x = train_x[idx]
        private_y = train_y[idx]
        private_dataset.append({"train_X":private_x, "train_Y":private_y,
                                "valid_X":valid_x, "valid_Y":valid_y,
                                "class":f'{private_classes[0]},{private_classes[1]},{private_classes[2]},{private_classes[3]},{private_classes[4]}'})
    test_x, public_x, test_y, public_y= train_test_split(testset.data, 
                                                         np.array(testset.targets), 
                                                         test_size=public_rate, 
                                                         stratify=np.array(testset.targets))
    public_dataset = {"X":public_x,"Y":public_y}
    test_dataset = {"X":test_x,"Y":test_y}

    return private_dataset, public_dataset, test_dataset
    
def Noniid_dataset_5class(N_client, public_rate=0.5):
    trainset, testset = get_cifar10()
    private_dataset = []
    classes = [i for i in range(10)]
    
    for n in range(N_client):
        if len(classes) == 0:
            classes = [i for i in range(10)]
        client_class = np.random.choice(len(classes),5, replace=False)
        client_class.sort()
        private_classes = [classes.pop(client_class[4]),classes.pop(client_class[3]),
                           classes.pop(client_class[2]),classes.pop(client_class[1]),classes.pop(client_class[0])]
        idx = [(c in private_classes) for c in trainset.targets]
        private_x = trainset.data[idx]
        private_y = np.array(trainset.targets)[idx]
        private_trn_x, private_val_x, private_trn_y, private_val_y= train_test_split(private_x, 
                                                                                     private_y, 
                                                                                     test_size=0.25,
                                                                                     stratify=private_y)
        private_dataset.append({"train_X":private_trn_x, "train_Y":private_trn_y,
                                "valid_X":private_val_x, "valid_Y":private_val_y,
                                "class":f'{private_classes[0]},{private_classes[1]},{private_classes[2]},{private_classes[3]},{private_classes[4]}'})
    
    test_x, public_x, test_y, public_y= train_test_split(testset.data, 
                                                         np.array(testset.targets), 
                                                         test_size=public_rate, 
                                                         stratify=np.array(testset.targets),
                                                         random_state=1)
                                                         
    public_dataset = {"X":public_x,"Y":public_y}
    test_dataset = {"X":test_x,"Y":test_y}

    return private_dataset, public_dataset, test_dataset


    
if __name__ == "__main__":
    train, public, test = Noniid_dataset_2class_uniformvalid(20,0.6,0.01)
    print(train[0]["train_X"].shape)
    print(train[0]["train_Y"].shape)
    print(train[0]["valid_X"].shape)
    print(train[0]["valid_Y"].shape)
    print(public["X"].shape)
    print(public["Y"].shape)
    print(test["X"].shape)
    print(test["Y"].shape)
    PATH = r'C:\Users\nckubot65904\Desktop\5\code\paper_implement\FedDyn-master\Data\CIFAR10_20_Dirichlet_0.600'
    datasets = np.load(os.path.join(PATH,'clnt_x.npy'))[0]
    labels = np.load(os.path.join(PATH,'clnt_y.npy'))[0]
    datasets = datasets.transpose((0,2,3,1))
    print(datasets.shape)
    print(labels.shape)
        