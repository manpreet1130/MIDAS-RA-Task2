'''
	This script consists of the Network class and the train and test functions
	used to train and test the CNN model. 
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


#The SummaryWriter is used to log data for consumption and visualization by Tensorboard
writer = SummaryWriter("runs/newDataTask1")

'''
	This network class in particular consists of 3 convolutional layers and 3 fully connected layers
	with 2 dropout layers added in between. The hyperparameters were selected after trying out many different
	models and choosing the ones which provided the best training accuracy. 
	The kernel size and padding were chosen as to not reduce the feature sizes to such an extent that no learning 
	takes place. 
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
		self.fc3 = nn.Linear(128, 62)

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

'''
	If GPU computation is possible, the tensors will be converted
	to CUDA tensor types, else will remain CPU tensors
'''
device = ("cuda" if torch.cuda.is_available() else "cpu")

model = Network()
model.to(device)


# All the images were converted to grayscale and resized to 28*28 before turning into tensors
transformations = transforms.Compose([
			transforms.Grayscale(),
			transforms.Resize((28, 28)),
			transforms.ToTensor()])

'''
	Loading in the required dataset from their folders and creating batches using DataLoader.
	A batch size of 128 was selected after trying out batch sizes 32, 64, 128 and 256. 
	It was seen that batch sizes 32 and 64 resulted in slower computational speeds, whereas 
	the batch size of 256 achieved lower accuracy in comparison to that of batch size 128.
'''
train_ds = ImageFolder(root = './train', transform = transformations)
#test_ds = ImageFolder(root = './data/test', transform = transformations)
#train_ds = MNIST(root = './data', train = True, download = True, transform = transformations)
test_ds = MNIST(root = './data', train = False, transform = transformations)
batch_size = 128
train_loader = DataLoader(train_ds, batch_size, shuffle = True)
test_loader = DataLoader(test_ds, batch_size, shuffle = False)


'''
	This block of code takes out one batch iteration of images and 
	displays them on the Tensorboard interface.
'''
examples = iter(train_loader)
example_data, example_target = examples.next()

image_grid = torchvision.utils.make_grid(example_data)
writer.add_image('images', image_grid)

'''
	The train function is used to train the network on the dataset provided.
	The cross entropy loss function is a widely used loss function for optimizing
	classification problems and was chosen for this reason.
	Adam optimizer was chosen as it is known to perform well with classification
	problems and the learning rate of 0.01 was used, which is default.
'''
def train():
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters())
	'''
	checkpoint = torch.load('./checkpoints/checkpoint100.pt')
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	loss = checkpoint['loss']
	'''
	
	model.train()

	n_epochs = 200
	epoch = 0
	total_step = len(train_loader)
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
			running_loss += loss.item()

		accuracy = 100.0 * correct / total
		print("Epoch: {} Accuracy: {} Loss: {}".format(epoch+1, accuracy, running_loss / len(train_loader)))

		#This block is used to add values to the training accuracy and training loss graphs in Tensorboard after each epoch
		writer.add_scalar('training loss', running_loss / len(train_loader), epoch)
		writer.add_scalar('training accuracy', accuracy, epoch)
		
		#Saves the model checkpoint after every 10 epochs
		if((epoch+1) % 10 == 0):
			print("Saving checkpoint...")
			torch.save({
				'epoch' : epoch,
				'model_state_dict' : model.state_dict(),
				'optimizer_state_dict' : optimizer.state_dict(),
				'loss' : loss,
				}, "./checkpoints/checkpoint" + str(epoch+1) + ".pt")

		epoch += 1

'''
	The test function is used to test the accuracy of the model once it has been trained. 
'''
test_acc = []
def test():
	checkpoint = torch.load('./checkpoints/checkpoint130.pt')
	model.load_state_dict(checkpoint['model_state_dict'])
	model.eval()
	for i, (images, labels) in enumerate(test_loader):
		images, labels = images.to(device), labels.to(device)
		#print(labels)
		preds = model(images)
		_, prediction = torch.max(preds, dim = 1)
		
		count = sum(prediction == labels).item()
		print(count / len(labels) * 1.0)
		test_acc.append(100.0 * count / len(labels))

	print(sum(test_acc) / len(test_acc))
	

	
train()
#test()
writer.close()

