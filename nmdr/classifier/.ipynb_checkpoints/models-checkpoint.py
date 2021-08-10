import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet1D(nn.Module):

    def __init__(self, n, num_classes, debug=False):
        super().__init__()
        self.debug = debug
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
        
        self.conv1 = nn.Conv1d(in_channels=1,
                               out_channels=64, 
                               kernel_size=3, 
                               stride=1)
        self.conv2 = nn.Conv1d(in_channels=64,
                       out_channels=128, 
                       kernel_size=3, 
                       stride=1)
        self.maxpool1 = nn.MaxPool1d(3, stride=2)
        self.conv3 = nn.Conv1d(in_channels=128,
                       out_channels=1, 
                       kernel_size=3, 
                       stride=1)
        self.maxpool2 = nn.MaxPool1d(3, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(297, num_classes)
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        if self.debug: print(x.shape)

        x = F.relu(self.conv2(x))
        if self.debug: print(x.shape)
            
        x = self.maxpool1(x)
        if self.debug: print(x.shape)       
        
        x = F.relu(self.conv3(x))
        if self.debug: print(x.shape)
        
        x = self.maxpool2(x)
        if self.debug: print(x.shape)
            
        x = self.dropout(x)
        if self.debug: print(x.shape)
            
        x = self.fc1(x)
        if self.debug: print(x.shape)
        
        # Remove unnecessary dimensions and change shape to match target tensor
        # [BATCH_SIZE, num_classes, 1, 1] --> [BATCH_SIZE,num_classes] to make predicitons
#         output = torch.squeeze(x)
        output = torch.reshape(x, (-1, self.num_classes))
        if self.debug: print("output:\t", output.shape)

        return output
    
class Network(nn.Module):

    def __init__(self, n, num_classes, debug=False):
        super().__init__()
        self.debug = debug
        self.fc1 = nn.Linear(n, 64)
        self.b1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64,128)
        self.b2 = nn.BatchNorm1d(128)
        self.d1 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(128,num_classes)

    def forward(self,x):

        x = F.relu(self.fc1(x))
        x = self.b1(x)
        x = F.relu(self.fc2(x))
        x = self.b2(x)
        x = self.d1(x)
        x = torch.sigmoid(self.fc3(x))
        
        # Remove unnecessary dimensions and change shape to match target tensor
        # [BATCH_SIZE, num_classes, 1, 1] --> [BATCH_SIZE,num_classes] to make predicitons
        output = torch.squeeze(x)
        if self.debug: print("output:\t", output.shape)

        return x