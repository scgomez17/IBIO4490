import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.layer1= nn.Sequential(
            nn.Conv2d(1, 150, kernel_size=3),
            nn.BatchNorm2d(150),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer2= nn.Sequential(
            nn.Conv2d(150, 75, kernel_size=3),
            nn.Dropout(0.15),
            nn.BatchNorm2d(75)
        )
        self.layer3= nn.Sequential(
            nn.Conv2d(75, 75, kernel_size=3),
            nn.Dropout(0.5),
            nn.BatchNorm2d(75),
        )
        self.layer4= nn.Sequential(
            nn.Conv2d(75, 150, kernel_size=3),
            nn.Dropout(0.15),
            nn.BatchNorm2d(150)
        )
        self.fc1= nn.Sequential(   
            nn.Linear(93750, 100),    
            nn.Dropout(0.15)
        )
        self.fc2= nn.Sequential(
            nn.Linear(100, 10)                
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
