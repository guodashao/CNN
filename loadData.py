#coding=utf8

'''
Created on 2017-3-6
@author:guoshun

this is a function that load and process matlab file
'''

from scipy.io import loadmat as load
import numpy as np
import matplotlib.pyplot as plt

def reformat(samples,labels):
	'''
	@samples:numpy array
	@labels:numpy array
	@new:numpy array
	@labels:numpy array
	'''
	new = np.transpose(samples,(3,0,1,2)).astype(np.float32)
	labels = np.array([x[0] for x in labels])
	one_hot_labels = []
	for i in labels:
		one_hot = [0.0] * 10
		if i != 10:
			one_hot[i] = 1
		else:
			one_hot[0] = 1
		one_hot_labels.append(one_hot)
	labels = np.array(one_hot_labels).astype(np.float32)
	return new,labels


def nomalize(samples):
	'''
	@samples: numpy array

	'''
	a = np.add.reduce(samples,keepdims=True,axis=3)
	a = a / 3.0
	return a/128 - 1

def inspect(dataset,labels,i):
	#show the image
	if dataset.shape[3] == 1:
		shape = dataset.shape
		dataset = dataset.reshape(shape[0],shape[1],shape[2])
	print labels[i]
	plt.imshow(dataset[i])
	plt.show()

def distribution(labels):
	#show the data distribution
	'''
	@labels:train_labels/test_labels
	'''
	count={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
	for label in labels:
		if label[0] == 1:
			count[label[0]] += 1
		elif label[0] == 2:
			count[label[0]] += 1
		elif label[0] == 3:
			count[label[0]] += 1
		elif label[0] == 4:
			count[label[0]] += 1
		elif label[0] == 5:
			count[label[0]] += 1
		elif label[0] == 6:
			count[label[0]] += 1
		elif label[0] == 7:
			count[label[0]] += 1
		elif label[0] == 8:
			count[label[0]] += 1
		elif label[0] == 9:
			count[label[0]] += 1
		else:
			count[0] += 1
	return count


num_labels = 10
image_size = 32
num_channels = 1

train = load('../data/train_32x32.mat')
test = load('../data/test_32x32.mat')

train_samples = train['X']
train_labels = train['y']
test_samples = test['X']
test_labels = test['y']

n_train_samples,_train_labels = reformat(train_samples,train_labels)	#the data for train
n_test_samples,_test_labels = reformat(test_samples,test_labels)		#the date for test

_train_samples = nomalize(n_train_samples)	#the data for train
_test_samples = nomalize(n_test_samples)	#the data for test

#_train_samples  ->   _train_labels
#_test_samples   ->   _test_labels

if __name__ == '__main__':
	print distribution(train_labels)
	print distribution(test_labels)
	print _train_labels
	inspect(n_train_samples,_train_labels,1000)
	inspect(_train_samples,_train_labels,1000)



