import torch 
import torch.nn
from PIL import Image
from torchvision import transforms
import os
import numpy as np

from torch.utils.data import Dataset




class custom_dataset(Dataset):

    def __init__(self, mode ="train", tr = None, image_path = "dataset/", label_path = "dataset/"):
        self.mode = mode 
        self.image_path = image_path

        self.tr = tr

        self.train_list = []
        self.train_labels = []

        self.test_list = []
        self.test_labels = []

        self.val_list = []
        self.val_labels = []

        self.image_list = []
        
        self.unique_labels = []
        count=0
        for i in os.listdir(image_path + "train/"):
            self.unique_labels.append(i)
            for j in os.listdir(image_path+"train/" + i):
                if count%3==0:            
                    self.train_list.append(i+'/'+j)
                    self.train_labels.append(i)
                count+=1

        for i in os.listdir(image_path + "test/"):            
            for j in os.listdir(image_path+"test/" + i):
                self.test_list.append(i+'/'+j)
                self.test_labels.append(i)

        for i in os.listdir(image_path + "val/"):            
                for j in os.listdir(image_path+"val/" + i):
                    if count%3==0:            
                        self.val_list.append(i+'/'+j)
                        self.val_labels.append(i)
                    count+=1

        if (self.mode == "train"):
            self.image_list = self.train_list
            self.labels = self.train_labels
        elif(self.mode == "test"):
            self.image_list =self.test_list
            self.labels = self.test_labels
        elif(self.mode == "val"):
            self.image_list =self.val_list
            self.labels = self.val_labels


    def __getitem__(self, index):

        image = Image.open(self.image_path+self.mode+'/' +self.image_list[index])
        # image.show()
        image.convert("RGB")
        label = self.labels[index]
        label = self.parse_labels(label)

        if(self.tr):
            image = self.tr(image)

        if self.mode == "train":
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) 
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
            
        image = transform(image)        


        if label is not None:
            label = torch.as_tensor(label)
        
        return image, label
    
    def parse_labels(self, label):
            arr = np.zeros((len(self.unique_labels),), dtype= float)
            for i in range(len(self.unique_labels)):
                if label == self.unique_labels[i]:
                    arr[i] = 1.0
            return arr
    
    def __len__(self):
        return len(self.image_list)
    

