import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # First layer sees: 32x32x3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, 
                    kernel_size=5, stride=1, padding=0)
        
        # Second layer sees: 28x28x16
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, 
                    kernel_size=5, stride=1, padding=0)
        
        # Third layer sees: 24x24x32
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, 
                    kernel_size=5, stride=1, padding=0)
        
        # This layer output 20 x 20 x 64
        
        self.fc1 = nn.Linear(20*20*64, 1000)
        self.fc2 = nn.Linear(1000, 46)
        self.dropout = nn.Dropout(p=0.25)
    
    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(-1, 20*20*64)
        
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x