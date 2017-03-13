#coding=utf8

'''
Created on 2017-3-6
@author:guoshun



the result is about:
(' Average  Accuracy:', 87.980769230769226)
('Standard Deviation:', 1.4039831334646624)

'''

import loadData
from dlMethod import Network

train_samples, train_labels = loadData._train_samples, loadData._train_labels
test_samples, test_labels = loadData._test_samples, loadData._test_labels

print('Training set', train_samples.shape, train_labels.shape)
print('    Test set', test_samples.shape, test_labels.shape)

image_size = loadData.image_size
num_labels = loadData.num_labels
num_channels = loadData.num_channels

net = Network(
		train_batch_size=64, test_batch_size=500, pooling_scale=2,
		dropout_rate = 0.9,
		base_learning_rate = 0.001, decay_rate=0.99)
net.define_inputs(
		train_samples_shape=(64, image_size, image_size, num_channels),
		train_labels_shape=(64, num_labels),
		test_samples_shape=(500, image_size, image_size, num_channels),
		)
#
net.add_conv(patch_size=3, in_depth=num_channels, out_depth=32, activation='relu', pooling=False, name='conv1')
net.add_conv(patch_size=3, in_depth=32, out_depth=32, activation='relu', pooling=True, name='conv2')
net.add_conv(patch_size=3, in_depth=32, out_depth=32, activation='relu', pooling=False, name='conv3')
net.add_conv(patch_size=3, in_depth=32, out_depth=32, activation='relu', pooling=True, name='conv4')

# 4 = 两次 pooling, 每一次缩小为 1/2
# 32 = conv4 out_depth
net.add_fc(in_num_nodes=(image_size // 4) * (image_size // 4) * 32, out_num_nodes=128, activation='relu', name='fc1')
net.add_fc(in_num_nodes=128, out_num_nodes=10, activation=None, name='fc2')

net.define_model()

net.run(train_samples,train_labels,test_samples,test_labels,3000)


