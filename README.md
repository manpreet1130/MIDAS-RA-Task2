# MIDAS-RA-Task2
This repository is a submission for the MIDAS Summer Internship/RA Task 2021.

## Table of Contents
1. [Task A](#task-a)
2. [Task B](#task-b)
3. [Task C](#task-c)
4. [References](#references)

## Task A
This folder consists of a **checkpoints** folder, and two python scripts by the name of **augment.py** and **cnnTaskA.py** respectively. 
1. The **augment.py** script comprises of three functions which were used to augment the dataset provided. These functions are aug(), zoom() and invert(). The aug() is responsible for adding rotation and translation to the images. The zoom() is responsible for warping the images. The invert() is used to convert the black alphanumeric characters to white and change the background from white to black.
2. The **cnnTaskA.py** script comprises of the Network class and two functions, train() and test(). This script is used to train the model on the dataset provided and the checkpoints are saved in the **checkpoints** folder.

## Task B
This folder comprises of folders by the names of **checkpointsMyData**, **checkpointsMyDataMNIST**, **checkpointsMyMNIST**, **results**, **runs** and a python script named **cnnTaskB.py**.
1. The **checkpointsMyData** folder has all the checkpoints used for training the network on only the 0-9 classes from the dataset provided in Task A. 
2. The **checkpointsMyDataMNIST** folder consists of all the checkpoints saved while training the pretrained network on the MNIST dataset.
3. The **checkpointsMyMNIST** folder consists of all the checkpoints saved while training a network from scratch on the MNIST dataset.
4. The **results** folder comprises of different graphs on training accuracy, training loss and confusion matrix for each network trained.
5. The **runs** folder is used to log information into Tensorboard. To access Tensorboard, `tensorboard --logdir=runs` can be executed from the terminal.
6. The **cnnTaskB.py** script is used for training and testing the different networks of this Task.

## Task C
This folder consists of folders by the names of **checkpointsPretrained**, **checkpointsScratch**, **results**, **runs** and two python scripts named **sortData.py** and **cnnTaskC.py**.
1. The **checkpointsPretrained** folder comprises of checkpoints saved while training the pretrained model from Task A on the dataset provided in this task.
2. The **checkpointsScratch** folder consists of checkpoints saved while training a model from scratch on the dataset provided.
3. The **results** folder has all the graphs for training accuracy and training loss of the two networks and also confusion matrix for both created when testing was done on the MNIST test split.
4. The **runs** folder is used to log information into Tensorboard. To access Tensorboard, `tensorboard --logdir=runs` can be executed from the terminal.
5. The **sortData.py** script is used to filter the images into their correct class folders. 
6. The **cnnTaskC.py** script is used for training and testing the networks of this Task.

## References
1. Image Augmentation: https://www.analyticsvidhya.com/blog/2019/12/image-augmentation-deep-learning-pytorch/
2. Dropout vs Batch Normalization: https://link.springer.com/article/10.1007/s11042-019-08453-9
3. Tensorboard: https://www.youtube.com/watch?v=VJW9wU-1n18
4. Confusion Matrix: https://deeplizard.com/learn/video/0LhiS6yu2qQ
