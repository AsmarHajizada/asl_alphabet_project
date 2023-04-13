import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from torchmetrics import F1Score
from torchmetrics.classification import MulticlassAccuracy


from model.vgg import vgg16
from model.resnet import ResNetModel
from dataset.dataset_retrieval import custom_dataset
from torch.optim import SGD, Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import os

save_model_path = "checkpoints/"
pth_name = "VGGAdam.pth" # change this to the name of the model you want to save




def val(model, data_val, loss_function, writer, epoch):

    f1 = F1Score(num_classes=29, task = 'multiclass', average='macro')
    accuracy = MulticlassAccuracy(num_classes=29)
    data_iterator = enumerate(data_val)  # take batches
    accuracy_list = []
    accuracyt_list = []
    f1_list = []
    f1t_list = []

    with torch.no_grad():
        model.eval()  # switch model to evaluation mode
        tq = tqdm(total=len(data_val))
        tq.set_description('Validation')

        total_loss = 0

        for _, batch in data_iterator:
            # forward propagation
            image, label = batch
            image = image.cuda()
            label = label.cuda()
            pred = model(image)
        
            loss = loss_function(pred, label)
            loss = loss.cuda()

            f1_list.extend(torch.argmax(pred, dim =1).tolist())
            f1t_list.extend(torch.argmax(label, dim =1).tolist())

            accuracy_list.extend(torch.argmax(pred, dim =1).tolist())
            accuracyt_list.extend(torch.argmax(label, dim =1).tolist())

            total_loss += loss.item()
            tq.update(1)

    tq.close()
    print("F1 score: ", f1(torch.tensor(f1_list), torch.tensor(f1t_list)))
    print("Accuracy: ", accuracy(torch.tensor(accuracy_list), torch.tensor(accuracyt_list)))

    writer.add_scalar("Validation mIoU", f1(torch.tensor(f1_list), torch.tensor(f1t_list)), epoch)
    writer.add_scalar("Validation Loss", total_loss/len(data_val), epoch)

    return None


def train(model, dataloader, val_loader, optimizer, loss_fn, n_epochs):
    device = 'cuda'
    writer = SummaryWriter()

    model.cuda()  # Move the model to the specified device (e.g., GPU or CPU)
    model.train()  # Set the model to training mode
    for epoch in range(n_epochs):
        running_loss = 0.0
        tq = tqdm(total=len(dataloader))
        tq.set_description('epoch %d' % (epoch))


        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)  # Move the batch of images to the specified device
            labels = labels.to(device)  # Move the batch of labels to the specified device
            
            optimizer.zero_grad()  # Reset the gradients of the optimizer
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = loss_fn(outputs, labels)
            outputs = outputs.softmax(dim=1)

            # Backward pass
            loss.backward()

            # Update model parameters
            optimizer.step()
            
            running_loss += loss.item()
            tq.set_postfix(loss_st='%.6f' % loss.item())
            tq.update(1)
            
        tq.close()

        epoch_loss = running_loss / len(dataloader)
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, n_epochs, epoch_loss))

        val(model, val_loader, loss_fn, writer, epoch)

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        torch.save(checkpoint, os.path.join(save_model_path, pth_name))
        print("saved the model " + save_model_path)
        model.train()

if __name__ == '__main__':
    train_data = custom_dataset("train")
    val_data = custom_dataset("val")

    train_loader = DataLoader(
        train_data,
        batch_size=8,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_data,
        batch_size=1,
        num_workers=4
    )

    model = ResNetModel(29).cuda()   
    model2 = vgg16(29).cuda()
    optimizer1 = SGD(model2.parameters(),  lr=0.002)  # change model and learning rate 
    optimizer2 = Adam(model2.parameters(), lr=0.0005) # change model and learning rate
    loss = nn.CrossEntropyLoss()


    train(model2, train_loader, val_loader, optimizer2, loss, 15) # change model, optimizer and epoch number 
    print('Finished Training: VGG16 Adam (lr=0.0001)') # change this to the name and parameters of the model you want to train