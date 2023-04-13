import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torchmetrics import F1Score
from torchmetrics.classification import MulticlassAccuracy

from torch.utils.data import DataLoader, Dataset
from model.vgg import vgg16
from model.resnet import ResNetModel
from dataset.dataset_retrieval import custom_dataset
from torch.optim import SGD, Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def test(model, data_test, loss_function, writer, epoch):
    f1 = F1Score(num_classes=29, task = 'multiclass', average='macro')
    accuracy = MulticlassAccuracy(num_classes=29)
    data_iterator = enumerate(data_test)  # take batches
    accuracy_list = []
    accuracyt_list = []
    f1_list = []
    f1t_list = []

    with torch.no_grad():
        model.eval()  # switch model to evaluation mode
        tq = tqdm(total=len(data_test))
        tq.set_description('Testing')

        total_loss = 0

        for _, batch in data_iterator:
            # forward propagation
            image, label = batch
            image = image.cuda()
            label = label.cuda()
            pred = model(image)
            probs = softmax(pred, dim=1)

            loss = loss_function(pred, label.float())
            loss = loss.cuda()

            pred = pred.softmax(dim=1)
            #confidence score 
            confidence_score = round(torch.max(probs, dim=1)[0].item(), 4)

            accuracy_list.extend(torch.argmax(pred, dim=1).tolist())
            accuracyt_list.extend(torch.argmax(label, dim=1).tolist())

            f1_list.extend(torch.argmax(pred, dim=1).tolist())
            f1t_list.extend(torch.argmax(label, dim=1).tolist())

            total_loss += loss.item()
            tq.update(1)

    tq.close()
    print("F1 score: ", f1(torch.tensor(f1_list), torch.tensor(f1t_list)))
    print("Accuracy: ", accuracy(torch.tensor(accuracy_list), torch.tensor(accuracyt_list)))
    print("Confidence score: ", confidence_score)
    writer.add_scalar("Validation mIoU", f1(torch.tensor(f1_list), torch.tensor(f1t_list)), epoch)
    writer.add_scalar("Validation Loss", total_loss/len(data_test), epoch)

    return None

if __name__ == '__main__':
    test_data = custom_dataset("test")

    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
    )

    model1 = ResNetModel(29).cuda()   # Initialsing an object of the class.
    model2 = vgg16(29).cuda()
    # optimizer = SGD(model2.parameters(),  lr=0.002)                   #change model and lr accordingly
    optimizer = Adam(model2.parameters(), lr=0.0001, weight_decay=1e-5) #change model and lr accordingly

    checkpoint = torch.load("checkpoints/VGGAdam.pth")

    model2.load_state_dict(checkpoint['state_dict'])    # change model accordingly
    optimizer.load_state_dict(checkpoint['optimizer'])
    loss = nn.CrossEntropyLoss()
    writer = SummaryWriter()

    test(model2, test_loader, loss, writer, 1)          # change model accordingly
    print('Finished Testing: VGG16 Adam (lr=0.0001)')   # change this to the model and optimizer you used