#coding=utf8

'''
Created on 2017-3-6
@author:guoshun

there are some neutral network methods
'''

import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

class Network():
	def __init__(self, train_batch_size, test_batch_size, pooling_scale,
				 dropout_rate, base_learning_rate, decay_rate,optimizeMethod='adam',
				 save_path='model/default.ckpt'):
		self.dropout_rate=dropout_rate
		self.base_learning_rate=base_learning_rate
		self.decay_rate = decay_rate
		self.train_batch_size = train_batch_size
		self.test_batch_size = test_batch_size
		self.optimizeMethod = optimizeMethod 

		# Graph Related
		self.tf_train_samples = None
		self.tf_train_labels = None
		self.tf_test_samples = None
		self.tf_test_labels = None

		


		# Hyper Parameters
		self.conv_config = []
		self.conv_weights = []
		self.conv_biases = []
		self.fc_config = []
		self.fc_weights = []
		self.fc_biases = []
		self.pooling_scale = pooling_scale
		self.pooling_stride = pooling_scale

		# 统计
		self.writer = None
		self.merged = None
		self.train_summaries = []
		#self.test_summaries = []



	def add_conv(self,patch_size,in_depth,out_depth,activation,pooling,name):
		'''
		patch_size:the window size
		out_depth:代表该层的神经元的个数
		in_depth:如果是第一层就代表颜色通道的数目,如果不是第一层代表上一层的out_depth
		'''
		self.conv_config.append({
			'patch_size': patch_size,
			'in_depth': in_depth,
			'out_depth': out_depth,
			'activation': activation,
			'pooling': pooling,
			'name': name
		})
		with tf.name_scope(name):
			weights = tf.Variable(tf.truncated_normal([patch_size,patch_size,in_depth,out_depth],stddev=0.1),name=name+'_weight')
			biases = tf.Variable(tf.constant(0.1,shape=[out_depth]),name=name+'_biase')
			self.conv_weights.append(weights)
			self.conv_biases.append(biases)


	def define_inputs(self,train_samples_shape, train_labels_shape, test_samples_shape):
		# 这里只是定义图谱中的各种变量
		with tf.name_scope('inputs'):
			self.tf_train_samples = tf.placeholder(tf.float32, shape=train_samples_shape, name='tf_train_samples')
			self.tf_train_labels = tf.placeholder(tf.float32, shape=train_labels_shape, name='tf_train_labels')
			self.tf_test_samples = tf.placeholder(tf.float32, shape=test_samples_shape, name='tf_test_samples')

	def add_fc(self,in_num_nodes, out_num_nodes, activation,name):
		self.fc_config.append({
			'in_num_nodes': in_num_nodes,
			'out_num_nodes': out_num_nodes,
			'activation': activation,
			'name': name
		})
		with tf.name_scope(name):
			weights = tf.Variable(tf.truncated_normal([in_num_nodes,out_num_nodes],stddev=0.1),name=name+'_weight')
			biases = tf.Variable(tf.constant(0.1,shape=[out_num_nodes]),name=name+'_biase')
			self.fc_weights.append(weights)
			self.fc_biases.append(biases)
			self.train_summaries.append(tf.histogram_summary(str(len(self.fc_weights))+'_weights', weights))
			self.train_summaries.append(tf.histogram_summary(str(len(self.fc_biases))+'_biases', biases))

	def apply_regularization(self, _lambda):
		# L2 regularization for the fully connected parameters
		regularization = 0.0
		for weights, biases in zip(self.fc_weights, self.fc_biases):
			regularization += tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)
		# 1e5
		return _lambda * regularization

	def train_data_iterator(self,samples, labels, iteration_steps, chunkSize):
		'''
		Iterator/Generator: get a batch of data
		这个函数是一个迭代器/生成器，用于每一次只得到 chunkSize 这么多的数据
		用于 for loop， just like range() function
		'''
		if len(samples) != len(labels):
			raise Exception('Length of samples and labels must equal')
		stepStart = 0  # initial step
		i = 0
		while i < iteration_steps:
			stepStart = (i * chunkSize) % (labels.shape[0] - chunkSize)
			yield i, samples[stepStart:stepStart + chunkSize], labels[stepStart:stepStart + chunkSize]
			i += 1

	def test_data_iterator(self,samples, labels, chunkSize):
		'''
		Iterator/Generator: get a batch of data
		这个函数是一个迭代器/生成器，用于每一次只得到 chunkSize 这么多的数据
		用于 for loop， just like range() function
		'''
		if len(samples) != len(labels):
			raise Exception('Length of samples and labels must equal')
		stepStart = 0  # initial step
		i = 0
		while stepStart < len(samples):
			stepEnd = stepStart + chunkSize
			if stepEnd < len(samples):
				yield i, samples[stepStart:stepEnd], labels[stepStart:stepEnd]
				i += 1
			stepStart = stepEnd

	def define_model(self):
		'''
		定义我的的计算图谱
		'''
		def model(data_flow,train=True):
			'''
			@data: original inputs
			@return: logits
			'''
			# Define Convolutional Layers
			for i, (weights, biases, config) in enumerate(zip(self.conv_weights, self.conv_biases, self.conv_config)):
				with tf.name_scope(config['name'] + '_model'):
					with tf.name_scope('convolution'):
						# default 1,1,1,1 stride and SAME padding
						data_flow = tf.nn.conv2d(data_flow, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
						data_flow = data_flow + biases
					if config['activation'] == 'relu':
						data_flow = tf.nn.relu(data_flow)
					else:
						raise Exception('Activation Func can only be Relu right now. You passed', config['activation'])
					if config['pooling']:
						data_flow = tf.nn.max_pool(
							data_flow,
							ksize=[1, self.pooling_scale, self.pooling_scale, 1],
							strides=[1, self.pooling_stride, self.pooling_stride, 1],
							padding='SAME')

			# Define Fully Connected Layers
			for i, (weights, biases, config) in enumerate(zip(self.fc_weights, self.fc_biases, self.fc_config)):
				if i == 0:
					shape = data_flow.get_shape().as_list()
					data_flow = tf.reshape(data_flow, [shape[0], shape[1] * shape[2] * shape[3]])
				with tf.name_scope(config['name'] + 'model'):

					### Dropout
					if i == len(self.fc_weights) - 1:
						data_flow =  tf.nn.dropout(data_flow, self.dropout_rate, seed=4926)
					###

					data_flow = tf.matmul(data_flow, weights) + biases
					if config['activation'] == 'relu':
						data_flow = tf.nn.relu(data_flow)
					elif config['activation'] is None:
						pass
					else:
						raise Exception('Activation Func can only be Relu or None right now. You passed', config['activation'])
			return data_flow
		# Training computation.
		logits = model(self.tf_train_samples)

		with tf.name_scope('train'):
			self.train_prediction = tf.nn.softmax(logits, name='train_prediction')
			#tf.add_to_collection("prediction", self.train_prediction)
		with tf.name_scope('test'):
			self.test_prediction = tf.nn.softmax(model(self.tf_test_samples), name='test_prediction')
			#tf.add_to_collection("prediction", self.test_prediction)

		with tf.name_scope('loss'):
			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self.tf_train_labels))
			self.loss += self.apply_regularization(_lambda=5e-4)
			self.train_summaries.append(tf.scalar_summary('Loss', self.loss))


		# learning rate decay
		global_step = tf.Variable(0)
		learning_rate = tf.train.exponential_decay(
			learning_rate=self.base_learning_rate,
			global_step=global_step*self.train_batch_size,
			decay_steps=100,
			decay_rate=self.decay_rate,
			staircase=True
		)

		# Optimizer.
		with tf.name_scope('optimizer'):
			if(self.optimizeMethod=='gradient'):
				self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
			elif(self.optimizeMethod=='momentum'):
				self.optimizer = tf.train.MomentumOptimizer(learning_rate, 0.5).minimize(self.loss)
			elif(self.optimizeMethod=='adam'):
				self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

		self.merged_train_summary = tf.merge_summary(self.train_summaries)
		#self.merged_test_summary = tf.merge_summary(self.test_summaries)


		# 放在定义Graph之后，保存这张计算图
		self.saver = tf.train.Saver(tf.all_variables())


	def run(self,train_samples,train_labels,test_samples,test_labels,iteration_steps):
		self.writer = tf.train.SummaryWriter('./board', tf.get_default_graph())
		with tf.Session(graph=tf.get_default_graph()) as session:
			tf.initialize_all_variables().run()

			### 训练
			print('Start Training')
			# batch 1000
			for i, samples, labels in self.train_data_iterator(train_samples, train_labels, iteration_steps=iteration_steps, chunkSize=self.train_batch_size):
				_, l, predictions, summary = session.run(
					[self.optimizer, self.loss, self.train_prediction, self.merged_train_summary],
					feed_dict={self.tf_train_samples: samples, self.tf_train_labels: labels}
				)
				self.writer.add_summary(summary, i)
				# labels is True Labels
				accuracy, _ = self.accuracy(predictions, labels)
				if i % 50 == 0:
					print('Minibatch loss at step %d: %f' % (i, l))
					print('Minibatch accuracy: %.1f%%' % accuracy)

			### 测试
			accuracies = []
			confusionMatrices = []
			for i, samples, labels in self.test_data_iterator(test_samples, test_labels, chunkSize=self.test_batch_size):
				result = session.run(
					self.test_prediction,
					feed_dict={self.tf_test_samples: samples}
				)
				#self.writer.add_summary(summary, i)
				accuracy, cm = self.accuracy(result, labels, need_confusion_matrix=True)
				accuracies.append(accuracy)
				confusionMatrices.append(cm)
				print('Test Accuracy: %.1f%%' % accuracy)
			print(' Average  Accuracy:', np.average(accuracies))
			print('Standard Deviation:', np.std(accuracies))

	def accuracy(self, predictions, labels, need_confusion_matrix=False):
		'''
		计算预测的正确率与召回率
		@return: accuracy and confusionMatrix as a tuple
		'''
		_predictions = np.argmax(predictions, 1)
		_labels = np.argmax(labels, 1)
		cm = confusion_matrix(_labels, _predictions) if need_confusion_matrix else None
		# == is overloaded for numpy array
		accuracy = (100.0 * np.sum(_predictions == _labels) / predictions.shape[0])
		return accuracy, cm


