import torch
import torch.nn as nn 
from torch.nn.functional import softmax

from torch.utils.data import DataLoader, Dataset 
from model.model import Model
from dataset.dataset_retrieval import custom_dataset
from torch.optim import SGD

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import metrics

import os

save_model_path = "checkpoints/"
pth_name = "saved_model.pth"


def val(model, data_val, loss_function, writer, epoch):
    

    f1_score = 0
    
    data_iterator = enumerate(data_val)     #take batches
    with torch.no_grad():

        model.eval()    #switch model to evaluation mode
        tq = tqdm(total=len(data_val))
        tq.set_description('Validation:')
        
        total_loss = 0

        for _, batch in data_iterator:
            #forward propagation
            image, label = batch
            pred = model(image.cuda())
            
            loss = loss_function(pred.cuda(), label.cuda())

            pred = pred.softmax(dim=1)
            f1_score += metrics(pred, label.cuda())

            total_loss += loss.item()
            tq.update(1)



    
    writer.add_scalar("Validation mIoU", f1_score/len(data_val), epoch)
    writer.add_scalar("Validation Loss", total_loss/len(data_val), epoch)
    
    print("F1 score: ", f1_score/len(data_val))

    tq.close()


    return None




def train (model, train_data, val_data,  optimizer, loss, max_epoch):

    writer = SummaryWriter()
    for epoch in range(max_epoch):
        
        model.train() # if you are going to update your model, put it in train mode.
        
        f1_score = 0 # to find total performance per epoch
        loss_total = 0

        data_iterator = enumerate(train_data)

        #tqdm is library to see he progressbar
        tq = tqdm(total=len(train_data)) 
        tq.set_description('epoch %d' % (epoch))

        for it, batch in data_iterator:
            optimizer.zero_grad()


            images, labels = batch

            pred = model(images.cuda())
            pred = softmax(pred, dim = 1)

            loss_value  =  loss(pred, labels.cuda())
            f1_score += metrics(pred, labels.cuda())

            loss_value.backward()
            optimizer.step(())

            loss_total +=loss_value.item() #pay attention!! if you dont write .item() you will overload gpu


            tq.set_postfix(loss_st='%.6f' % loss_value)
            tq.update(1)

        writer.add_scalar("Training F1", f1_score/len(train_data), epoch)
        writer.add_scalar("Validation Loss", loss_total/len(val_data), epoch)
            
        tq.close()

        val(model, val_data, loss, writer, epoch)



        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        torch.save(checkpoint, os.path.join(save_model_path, pth_name))
        print("saved the model " + save_model_path)


            


train_data = custom_dataset("train")
test_data = custom_dataset("test")

train_loader = DataLoader(
    train_data,
    batch_size=2,
    shuffle=True
)

iterator = enumerate(train_loader)

_, batch = next(iterator)
image, label = batch

test_loader = DataLoader(
    test_data,
    batch_size = 1
)

model = Model(28).cuda()

optimizer = SGD(model.parameters(),  lr = 0.001)

loss = nn.CrossEntropyLoss()

train(model, train_data, test_data,  optimizer, loss, max_epoch=10)

# print(model.forward(torch.randn(1, 3, 256, 256)))


print(custom_dataset().unique_labels)


# if you want to load your pretrained model or
# you want to resume stopped training
# use torch.load_state_dict by checking the library!


