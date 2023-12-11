
#----------------------

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os


import pandas as pd
from PIL import Image
#---------------------
class ImageDataset(Dataset):
    def __init__(self, file_list, img_dir, transform=None):
        self.file_list = file_list
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.file_list[idx])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Extract the age from the filename assuming the age is prefixed
        age = int(self.file_list[idx].split('_')[0])
        age = torch.tensor(age, dtype=torch.float32)

        return image, age

                                                                                         
#---------------------


image_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
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


"""
class MultiModalCNN(nn.Module):
    def init(self, num_feature_inputs, output_dim):
        super(MultiModalCNN, self).init()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
  
        self.fc_img = nn.Linear(64 * 32 * 32, 256) 

   
        self.fc_features = nn.Linear(num_feature_inputs, 256)

        self.fc_combined = nn.Linear(256 * 2, 512) 
        self.fc_final = nn.Linear(512, output_dim) 

    def forward(self, image, features):
        # Image layers
        x1 = self.pool(F.relu(self.conv1(image)))
        x1 = self.pool(F.relu(self.conv2(x1)))
        x1 = x1.view(-1, 64 * 32 * 32)  # Flatten the output
        x1 = F.relu(self.fc_img(x1))

        x2 = F.relu(self.fc_features(features))

        # Combine features from both image and tabular data
        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fc_combined(x))
        x = self.fc_final(x)
        return x
"""                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         