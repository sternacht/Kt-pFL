import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, shufflenet_v2_x1_0, mobilenet_v3_large, alexnet, googlenet
from torchvision.models import ResNet50_Weights, ResNet18_Weights, ShuffleNet_V2_X1_0_Weights, MobileNet_V3_Large_Weights, AlexNet_Weights, GoogLeNet_Weights

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(64*8*8,512),
            nn.ReLU()
        )
        self.output = nn.Linear(512,10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)

        output = self.output(x)
        return output

    
class Costum_nlayer_CNN(nn.Module):
    def __init__(self, c=(128,128,192), num_classes=10, drop_rate=0.2):
        super(Costum_nlayer_CNN, self).__init__()
        self.layernum = len(c)
        if self.layernum == 2:
            c1,c2 = c
        elif self.layernum == 3:
            c1,c2,c3 = c
        elif self.layernum == 4:
            c1,c2,c3,c4 = c
        else:
            ValueError('Wrong num of layers. It should be 2, 3 or 4')
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=c1,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.Dropout2d(p=drop_rate),
            # nn.ZeroPad2d(padding=(1,0,1,0)),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(c1,c2,3,1,padding="same"),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.Dropout2d(p=drop_rate),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        if self.layernum > 2:
            self.conv3 = nn.Sequential(
                nn.Conv2d(c2,c3,3,1,padding="same"),
                nn.BatchNorm2d(c3),
                nn.ReLU(),
                nn.Dropout2d(p=drop_rate),
                nn.MaxPool2d(kernel_size=2),
            )
        if self.layernum > 3:
            self.conv4 = nn.Sequential(
                nn.Conv2d(c3,c4,3,1,padding="same"),
                nn.BatchNorm2d(c4),
                nn.ReLU(),
                nn.Dropout2d(p=drop_rate),
                nn.MaxPool2d(kernel_size=2),
            )
        w = int(32/(2**self.layernum))
        self.output = nn.Linear(c[-1]*w*w,num_classes)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.layernum > 2:
            x = self.conv3(x)
        if self.layernum > 3:            
            x = self.conv4(x)
        x = x.view(x.size(0),-1)
        output = self.output(x)
        return output
    
class Resnet18(nn.Module):
    def __init__(self, num_classes):
        super(Resnet18, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes)
        )
    
    def get_name():
        return 'Resnet18'
    
    def forward(self,x):
        x = self.model(x)
        # x = x.view(x.size(0),-1)
        return x

class Resnet50(nn.Module):
    def __init__(self, num_classes):
        super(Resnet50, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes)
        )
    
    def get_name():
        return 'Resnet50'

    def forward(self,x):
        x = self.model(x)
        # x = x.view(x.size(0),-1)
        return x


class ShuffleNet(nn.Module):
    def __init__(self, num_classes):
        super(ShuffleNet, self).__init__()
        self.model = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.DEFAULT)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes)
        )
    
    def get_name():
        return 'Shufflenet_v2x1'
    
    def forward(self,x):
        x = self.model(x)
        return x

class Mobilenet_v3(nn.Module):
    def __init__(self, num_classes):
        super(Mobilenet_v3, self).__init__()
        self.model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        # num_ftrs = self.model.
        # new_fc_layer = nn.Linear(in_features=self.model.classifier[3].in_features, out_features=num_classes)

        self.model.classifier[3] = nn.Sequential(
            nn.Linear(in_features=self.model.classifier[3].in_features, out_features=num_classes)
        )
    
    def get_name():
        return 'Mobilenet_v3'
    
    def forward(self,x):
        x = self.model(x)
        return x

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)  ##本来是16*5*5，mnist改为16*4*4
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def get_name():
        return 'LeNet'
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

########AlexNet#######
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def get_name():
        return 'AlexNet'

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x



if __name__ == '__main__':
    import torch
    import numpy as np
    model = LeNet(num_classes=10)
    test = np.zeros((3,32,32)).astype(np.float32)
    test = torch.from_numpy(test)
    output = model(test)