from itertools import groupby, cycle, tee, islice, chain, izip
import numpy as np
import random
import tensorflow as tf

n_windows = 3 
# n_vocabSize = len(DICT)
# n_embedding = len(embedding_dict['gandhi'][0])
n_vocabSize = 50
n_embedding = 11
n_wordDepth = n_windows*n_embedding
n_maxPool = 300
n_semanticSpace = 128
n_unclickedDocs = 4
n_filters = 1

#######################################################################################
"""
tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
Computes a 2-D convolution given 4-D input and filter tensors.
** value: A 4D Tensor. Must be of type float32 or float64 |||||| [batch, in_height, in_width, in_channels]
** filters: A 4D Tensor. Must have the same type as input |||||| [filter_height, filter_width, in_channels, out_channels]
batch = no. of examples
in_height = 1 for text CNN
in_width = embedding size
in_channels = no. of tokens
out_channels = n_maxPool
filter_height = 1 for text CNN
filter_width = embedding size
query = tf.placeholder(tf.int32, [None, n_wordDepth], name="query")
"""
#######################################################################################
learning_rate = 0.01
NEG = 50
batch_size = 3
Q_sentence_size = 15
D_sentence_size = 15
filter_size = 2
filter_shape = [1, filter_size, n_embedding, n_maxPool]
semantic_filter_shape = [n_maxPool, n_semanticSpace]
graph = tf.Graph()
with graph.as_default():
	"""QUERY"""
	Q_input_shape = [batch_size, Q_sentence_size, n_embedding]
	query = tf.placeholder(tf.float32, Q_input_shape, name="query")
	_query = tf.expand_dims(query, -3)
	# Convolution Layer --> Apply nonlinearity --> Maxpooling over the outputs --> "query semantic vector representation"
	W_c = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_c")
	b_c = tf.Variable(tf.constant(0.1, shape=[n_maxPool]), name="b_c")
	Q_conv = tf.nn.conv2d(_query, W_c,strides=[1, 1, 1, 1],padding="VALID",name="conv")
	Q_h = tf.nn.tanh(tf.nn.bias_add(Q_conv, b_c), name="tanh")
	Q_pooled = tf.nn.max_pool(Q_h, ksize=[1, 1, Q_sentence_size - filter_size + 1, 1], strides=[1, 1, 1, 1],padding='VALID',name="pool")
	Q_h_pool_flat = tf.reshape(Q_pooled, [batch_size, n_maxPool])	
	W_s0 = tf.Variable(tf.truncated_normal(semantic_filter_shape, stddev=0.1), name="W_s0")
	b_s0 = tf.Variable(tf.constant(0.1, shape=[n_semanticSpace]), name="b_s0")
	query_sem = tf.nn.xw_plus_b(Q_h_pool_flat, W_s0, b_s0)
	"""DOC"""
	D_input_shape = [batch_size, D_sentence_size, n_embedding]
	doc = tf.placeholder(tf.float32, D_input_shape, name="doc")
	_doc = tf.expand_dims(doc, -3)
	# Convolution Layer --> Apply nonlinearity --> Maxpooling over the outputs --> "query semantic vector representation"
	W_d = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_d")
	b_d = tf.Variable(tf.constant(0.1, shape=[n_maxPool]), name="b_d")
	D_conv = tf.nn.conv2d(_doc, W_d,strides=[1, 1, 1, 1],padding="VALID",name="conv")
	D_h = tf.nn.tanh(tf.nn.bias_add(D_conv, b_d), name="tanh")
	D_pooled = tf.nn.max_pool(D_h, ksize=[1, 1, D_sentence_size - filter_size + 1, 1], strides=[1, 1, 1, 1],padding='VALID',name="pool")
	D_h_pool_flat = tf.reshape(D_pooled, [batch_size, n_maxPool])	
	W_s1 = tf.Variable(tf.truncated_normal(semantic_filter_shape, stddev=0.1), name="W_s1")
	b_s1 = tf.Variable(tf.constant(0.1, shape=[n_semanticSpace]), name="b_s1")
	doc_sem = tf.nn.xw_plus_b(D_h_pool_flat, W_s1, b_s1)
	"""Calculations"""
	def calculation(query_sem, doc_sem, NEG, batch_size):
		temp = tf.tile(doc_sem, [1, 1])
		for i in range(NEG):
			rand = int((random.random() + i) * batch_size / NEG)
			doc_sem = tf.concat(0,[doc_sem,tf.slice(temp, [rand, 0], [batch_size - rand, -1]),tf.slice(temp, [0, 0], [rand, -1])])

		query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(query_sem), 1, True)), [NEG + 1, 1])
		doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_sem), 1, True))
		prod = tf.reduce_sum(tf.mul(tf.tile(query_sem, [NEG + 1, 1]), doc_sem), 1, True)
		norm_prod = tf.mul(query_norm, doc_norm)
		cos_sim_raw = tf.truediv(prod, norm_prod)
		cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [NEG + 1, batch_size]))*20
		prob = tf.nn.softmax((cos_sim))
		return prob

	prob = calculation(query_sem, doc_sem, NEG, batch_size)
	hit_prob = tf.slice(prob, [0, 0], [-1, 1])
	loss = -tf.reduce_sum(tf.log(hit_prob)) / batch_size

	'''training'''
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

	'''TEST'''
	test_query = tf.placeholder(tf.float32, shape=[1, sentence_size, n_embedding], name="test_query")



with tf.Session(graph=graph) as session:
	init = tf.initialize_all_variables()
	session.run(init)
	print('Initialized')
	sample_size = 10
	Qs = []
	Ds = []
	for i in range(sample_size):
		Q = np.random.rand(batch_size, Q_sentence_size, n_embedding)
		Qs.append(Q)
		D = np.random.rand(batch_size, D_sentence_size, n_embedding)
		Ds.append(D)
	feed_dict = {query: Qs[0], doc: Ds[0]}
	q = session.run([train_step], feed_dict=feed_dict)

