import torch 
import torch.nn
from PIL import Image
from torchvision import transforms
import os

from torch.utils.data import Dataset




class custom_dataset(Dataset):

    # initialize your dataset class
    def __init__(self, mode ="train", tr = None, image_path = "dataset/", label_path = "dataset/"):
        self.mode = mode 
        self.image_path = image_path

        self.tr = tr

        self.train_list = []
        self.train_labels = []

        self.test_list = []
        self.test_labels = []

        self.unique_labels = []

        for i in os.listdir(image_path + "train/"):
            self.unique_labels.append(i)
            for j in os.listdir(image_path+"train/" + i):
                self.train_list.append(i+'/'+j)
                self.train_labels.append(i)

        for i in os.listdir(image_path + "test/"):  
            for j in os.listdir(image_path+"test/" + i):
                self.test_list.append(i+'/'+j)
                self.test_labels.append(i)

        if (self.mode == "train"):
            self.image_list = self.train_list
            self.labels = self.train_labels
        elif(self.mode == "test"):
            self.image_list =self.test_list
            self.labels = self.test_labels



    def __getitem__(self, index):
        # getitem is required field for pytorch dataloader. Check the documentation

        image = Image.open(self.image_path+self.mode+'/' +self.image_list[index])
        image.show()
        label = self.labels[index]
        label = self.parse_labels(label)

        if(self.tr):
            image = self.tr(image)

        image = transforms.Resize((224, 224), interpolation = transforms.InterpolationMode.NEAREST)(image)
        
        transform = transforms.Compose([transforms.ToTensor()])

        image = transform(image)

        label = torch.as_tensor(label)

        print(label)
        return image, label
    

    def parse_labels(self, label):
        for i in range(len(self.unique_labels)):
            if label == self.unique_labels[i]:
                return i


    
    def __len__(self):
        return len(self.image_list)
    


cd = custom_dataset()[20]

