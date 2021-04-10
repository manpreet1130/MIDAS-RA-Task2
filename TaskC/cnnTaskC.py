'''
    This script is used to train the networks required in Task C. One of the networks
    uses the pretrained model from Task A and has the number of classes as 62, whereas 
    another is to be trained from scratch and the number of classes used then was 10. 
    While training both of these, the number of classes in the Network class was changed 
    manually. 
    All other functions, variables and hyperparameters are the same as used in the cnnTaskB.py
    script.
'''
import torch
import torchvision
from torchvision.datasets import ImageFolder, MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools

writer = SummaryWriter("runs/newCheckpointsTask3")

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size = 3, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 3, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 3, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.dropout1 = nn.Dropout(p = 0.25)
        self.dropout2 = nn.Dropout(p = 0.50)
        self.fc1 = nn.Sequential(
            nn.Linear(5*5*256, 256),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU())
        self.fc3 = nn.Linear(128, 10)

    def forward(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.dropout1(out)
        out = out.reshape(out.shape[0], -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.dropout2(out)
        out = F.softmax(self.fc3(out), dim = 1)
        return out

device = ("cuda" if torch.cuda.is_available() else "cpu")

model = Network()
model.to(device)

transformations = transforms.Compose([
                    transforms.Grayscale(),
                    transforms.Resize((28, 28)),
                    transforms.ToTensor()])

train_ds = ImageFolder(root = '/home/manpreet/MIDAS/TaskC/sortedData', transform = transformations)
#train_ds = MNIST(root = './data', train = True, download = True, transform = transformations)
test_ds = MNIST(root = './data', train = False, transform = transformations)
batch_size = 128
train_loader = DataLoader(train_ds, batch_size, shuffle = True)
test_loader = DataLoader(test_ds, batch_size, shuffle = False)

examples = iter(train_loader)
example_data, example_target = examples.next()

image_grid = torchvision.utils.make_grid(example_data)
writer.add_image('images', image_grid)


def train():
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    checkpoint = torch.load('/home/manpreet/MIDAS/TaskA/checkpoints/checkpoint130.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    

    model.train()


    n_epochs = 100
    epoch = 0
    #acc_list = []
    while epoch != n_epochs:
        total = 0
        correct = 0
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(images)
            #_, predictions = torch.max(preds, dim = 1)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            total += labels.nelement()
            #total = labels.size(0)
            _, predicted = torch.max(preds.data, 1)
            correct += (predicted == labels).sum().item()
            #acc_list.append(correct / total)
            running_loss += loss.item()
        
        accuracy = 100.0 * correct / total
        print("Epoch: {} Accuracy: {} Loss: {}".format(epoch + 1, accuracy, running_loss / len(train_loader)))
        writer.add_scalar('training loss', running_loss / len(train_loader), epoch)
        writer.add_scalar('training accuracy', accuracy, epoch)
        
        if((epoch+1) % 10 == 0):
            print("Saving checkpoint...")
            torch.save({
            'epoch' : epoch,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'loss' : loss,
            }, "/home/manpreet/MIDAS/TaskA/checkpoints/checkpoint" + str(epoch+1) + ".pt")
        
        epoch += 1

test_acc = []

def test(createConfusionMatrix = 0):
    checkpoint = torch.load('/home/manpreet/MIDAS/TaskC/checkpointsScratch/checkpoint30.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    if(createConfusionMatrix):
        confusionMatrix = torch.zeros(10, 10, dtype = torch.int64).to(device)

    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        #print(labels)
        preds = model(images)
        _, predictions = torch.max(preds, dim = 1)
        #if(labels == 9): print(predictions.item())
        if(createConfusionMatrix):
            for prediction, target in zip(predictions, labels):
                confusionMatrix[target.item(), prediction.item()] += 1
            
            
        count = sum(predictions == labels).item()
        test_acc.append(100.0 * count / len(labels))
    
    print("Testing Accuracy: {}".format(sum(test_acc) / len(test_acc)))
    
    if(createConfusionMatrix):
        classes = []
        for i in range(10):
                classes.append(str(i))
        plotConfusionMatrix(confusionMatrix, classes)
    

def plotConfusionMatrix(cm, classes, title = 'Confusion Matrix', cmap = plt.cm.Blues):
    print(cm)
    plt.imshow(cm.to("cpu"), interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment = "center", color = "white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    plt.show()


#train()
#writer.close()
test()