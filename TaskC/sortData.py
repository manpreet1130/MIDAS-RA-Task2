'''
    The dataset provided in Task C comprised of images distributed amongst 10 folders, 
    named 0-9. These images were jumbled and to place correct images under the folder with
    their correct label, this script was used.
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
from torchvision.utils import save_image
import random
import sys
import cv2

'''
    The Network class is the CNN used to train networks for tasks given in subpart 2.
    Using the best model obtained from subpart 2, the unsorted data is sorted.
'''
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
'''
    Loading in the dataset provided as a test dataset as predictions on the images will
    be used to place the images under their correct label. 
'''
test_ds = ImageFolder(root = './mnistTask/', transform = transformations)
batch_size = 1
test_loader = DataLoader(test_ds, batch_size, shuffle = False)

# This counter is used to maintain the image name format under each class
counter = {
    0 : 0,
    1 : 0,
    2 : 0,
    3 : 0,
    4 : 0,
    5 : 0,
    6 : 0,
    7 : 0,
    8 : 0,
    9 : 0 
}

'''
    This function is used in sorting the data. It was observed that a specific folder consisted of 
    no images of that label. For eg, the folder 0 consisted no images of 0's. This observation was used
    to further increase the accuracy of filtering the images.
    The best training accuracy in Task B was obtained using the pretrained network further trained on the MNIST 
    dataset. Hence, this network was loaded in to make predictions on this dataset.
'''
def test():
    checkpoint = torch.load('/home/manpreet/MIDAS/TaskB/checkpointsMyDataMNIST/checkpoint60.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)

        preds = model(images)
        _, prediction = torch.max(preds, dim = 1)
        folder = prediction.item()
        
        if(counter[folder] % 2500 == 0):
            print("Number: {}, Counter: {}".format(prediction.item(), counter[folder]))
        
        if(folder != labels.item()):
            save_image(images, './sortedData/' + str(folder) + '/' + str(counter[folder]) + '.jpg')
        count = sum(prediction == labels).item()
        counter[folder] += 1

test()