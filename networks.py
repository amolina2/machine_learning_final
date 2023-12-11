import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

#------------------------------------------------------------

class AgePredictionCNN(nn.Module):
    def __init__(self):
        super(AgePredictionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(64 * 56 * 56, 128)  
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


#-------------------------------------

      
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalCNN(nn.Module):
    def init(self, num_features, num_classes):
        super(MultiModalCNN, self).init()
    
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

       
        self.fc1_img = nn.Linear(128 * 32 * 32, 512)  

      
        self.fc1_tab = nn.Linear(num_features, 256)

      
        self.fc2_combined = nn.Linear(512 + 256, num_classes)

    def forward(self, x_img, x_tab):
        
        x_img = self.pool(F.relu(self.conv1(x_img)))
        x_img = self.pool(F.relu(self.conv2(x_img)))
        x_img = self.pool(F.relu(self.conv3(x_img))) 
        x_img = x_img.view(-1, 128 * 32 * 32) 
        x_img = F.relu(self.fc1_img(x_img))
        x_img = self.dropout(x_img)  

    
        x_tab = F.relu(self.fc1_tab(x_tab))

      
        x = torch.cat((x_img, x_tab), dim=1)
        x = F.relu(self.fc2_combined(x))
        return x

#-----------------------------------------------------------------