from torch import nn, Tensor
import torch.nn.functional as F

class Classifier(nn.Module):
    """
    Simple model with linear layers for CIFAR10
    """

    def __init__(self):
        super(Classifier, self).__init__()
        
        
        self.fc1 = nn.Linear(3*32*32, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(128, 10)
        # self.out_act = nn.Softmax(dim=1)

        

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.out(x)
        # x = self.out_act(x)
        return x


        
