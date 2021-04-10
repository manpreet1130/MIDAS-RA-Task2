'''
	This script is used to augment the dataset images provided. Data augmentation is done as the 
	deep learning model requires a sufficient amount of data to learn and the initial 40 images
	per class weren't enough. Using this script, the number of images per class were increased
	from 40 to 160.
	Further, the alphanumerics were turned to white and the background was changed to black. This
	was done as the MNIST dataset consists of white numerals on a black background and the dataset
	provided for the last part also follows the same pattern.
'''
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import skimage.io as io
from skimage.transform import rotate, AffineTransform, warp
import matplotlib.pyplot as plt
import cv2
import random
import os

'''
	This function is used to move across all folders in the dataset and randomly apply either a rotation,
	a translation or both to the images. 
	The rotation degree may vary from -15 to 15 degrees, both inclusive.
	The translation varies from -10 pixels to 10 pixels in both the x-axis and the y-axis.
	Using this function, the 40 images in each class were increased to 80.
'''
def aug():
	for i in range(1, 63):
		count = 56
		if(i < 10):
			directory = '/home/manpreet/MIDAS/TaskA/train/Sample00' + str(i)
		else:	
			directory = '/home/manpreet/MIDAS/TaskA/train/Sample0' + str(i)

		for filename in os.listdir(directory):
			if filename.endswith(".png"):
				image = io.imread(directory + '/' + filename)

				choice = random.randint(1, 3)
				angle = random.randint(-15, 15)
				shift = random.randint(-10, 10)
				while(angle == 0): angle = random.randint(-15, 15)
				while(shift == 0): shift = random.randint(-10, 10)
				transform = AffineTransform(translation = (shift, shift))

				if(choice == 1):
					image = rotate(image, angle = angle, mode = 'wrap')
				elif(choice == 2):
					image = warp(image, transform, mode = 'wrap')
				else:
					image = rotate(image, angle = angle, mode = 'wrap')
					image = warp(image, transform, mode = 'wrap')
				
				if(count < 100): name = '/img0' + str(i) + '-0'
				else: name = '/img0' + str(i) + '-'
				io.imsave(directory + name + str(count) + '.png', image)
				count += 1
			else: continue

'''
	This function is responsible for warping the images further, which will add another augmentation
	element to the dataset given. The zoom was done by 20% in both the x and y directions. This warp function
	was added to increase the dataset and inturn increase the accuracy. The 80 images per class achieved after applying
	the aug function were now increased to 160 after this function was applied.
'''
def zoom():
	for i in range(11, 63):
		count = 96
		if(i >= 10): directory = '/home/manpreet/MIDAS/TaskA/train/Sample0' + str(i)
		else: directory = '/home/manpreet/MIDAS/TaskA/train/Sample00' + str(i)
		for filename in os.listdir(directory):
			if filename.endswith(".png"):
				image = cv2.imread(directory + '/' + filename)
				height, width, channels = image.shape
				percent1 = 0.10
				percent2 = 0.90

				pts1 = np.float32([[width*percent1, height*percent1], [width*percent2, height*percent1],
									[width*percent1, height*percent2], [width*percent2, height*percent2]])
				pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
				M = cv2.getPerspectiveTransform(pts1, pts2)
				dst = cv2.warpPerspective(image, M, (width, height))
				if(count < 100): name = '/img0' + str(i) + '-0'
				else: name = '/img0' + str(i) + '-'
				cv2.imwrite(directory + name + str(count) + '.png', dst)
				count += 1
			else: continue

'''
	It was observed that the dataset provided consisted of black alphanumerics on a white background. 
	The MNIST dataset and the dataset provided in Task C both consisted of white numerals on black backgrounds,
	and hence, this dataset was also inverted to match that. 
'''
def invert():
	for i in range(1, 63):
		if(i >= 10): directory = '/home/manpreet/MIDAS/TaskA/train/Sample0' + str(i)
		else: directory = '/home/manpreet/MIDAS/TaskA/train/Sample00' + str(i)
		for filename in os.listdir(directory):
			if filename.endswith(".png"):
				image = cv2.imread(directory + '/' + filename)
				image = cv2.bitwise_not(image.copy(), image.copy())
				cv2.imwrite(directory + '/' + filename, image)

if __name__ == "__main__":
	invert()
	aug()
	zoom()