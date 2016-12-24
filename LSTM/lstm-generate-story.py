from __future__ import print_function
import gensim
from gensim.models  import Word2Vec
import os
import sys
import nltk
import random
import string
import zipfile
import numpy as np
import pandas as pd
import tensorflow as tf

global story
global WV
global word

def prepare_sentences(data):
	words = nltk.word_tokenize(data)
	sentence = []
	sentences = []
	text = []
	for word in words:
		if word =='.':
			sentences.append(sentence)
			sentence = []
		else:
			text.append(word)
			sentence.append(word)
	global story
	story = list(text)
	return sentences
def word_embedding(fname):
	with open(fname, 'r') as the_file:
		data = the_file.read()
	sentences = prepare_sentences(data)
	model = Word2Vec(sentences, min_count=1)
	dic = np.array(model.vocab.keys())
	cols = []
	vocab_vec = np.zeros(len(model[dic[0]]))
	for word in dic:
		word_vec = model[word]
		vocab_vec = np.vstack((vocab_vec, word_vec))
		cols = np.append(cols, word)
	vocab_vec = np.delete(vocab_vec, (0), axis=0)
	vocab_vec = vocab_vec.transpose()
	index = [i for i in range(len(model[dic[0]]))]
	df = pd.DataFrame(vocab_vec, index=index, columns=cols)
	return df
def word2embedding(token):
	return word.get(token)
def embedding2word(dictid):
	for key, ide in word.items():
		if ide == dictid:
			return key
class GenerateBatch(object):
  def __init__(self, text, batch_size, num_unrollings):
	self.text = text
	self.text_size = len(text)
	self.batch_size = batch_size
	self.num_unrollings = num_unrollings
	segment = self.text_size // batch_size
	self.cursor = [ i * segment for i in range(batch_size)]
	self.last_batch = self._next_batch()
  def _next_batch(self):
	batch = np.zeros(shape=(self.batch_size, vocabulary_size), dtype=np.float)
	for b in range(self.batch_size):
	  batch[b, word2embedding(self.text[self.cursor[b]])] = 1.0
	  self.cursor[b] = (self.cursor[b] + 1) % self.text_size
	return batch
  
  def next(self):
	batches = [self.last_batch]
	for step in range(self.num_unrollings):
	  batches.append(self._next_batch())
	self.last_batch = batches[-1]
	return batches
def words(probabilities):
	x = [embedding2word(c) for c in np.argmax(probabilities, 1)]
	return x
def batches2string(batches):
	s = ['']*batches[0].shape[0]
	for b in batches:
		s = [' '.join(x) for x in zip(s, words(b))]
	return s
def logprob(predictions, labels):
	predictions[predictions < 1e-10] = 1e-10
	return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]
def sample_distribution(distribution):
	r = random.uniform(0, 1)
	s = 0
	for i in range(len(distribution)):
		s += distribution[i]
		if s >= r:
			return i
	return len(distribution) - 1
def sample(prediction):
	p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
	k = sample_distribution(prediction[0])
	p[0, k] = 1.0
	return p
def random_distribution():
	b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
	x = b/np.sum(b, 1)[:,None]  
	return x 

WV = word_embedding("gandhi.txt")
valid_size = 25
valid_text = story[:valid_size]
train_text = story[valid_size:]
train_size = len(story)
embedding_size, vocabulary_size = WV.shape
word = {}
for i in range(vocabulary_size):
	word[WV.columns[i]]=i
batch_size = 25
num_unrollings = 5
train_batches = GenerateBatch(train_text, batch_size, num_unrollings)
valid_batches = GenerateBatch(valid_text, 1, 1)

num_nodes = 80
graph = tf.Graph()
with graph.as_default():
  	# Parameters:
	vocabulary_embeddings = tf.Variable(tf.cast(WV.as_matrix().transpose(), tf.float32), name="vocabulary_embeddings")
	sx = tf.Variable(tf.truncated_normal([embedding_size, num_nodes*4], -0.1, 0.1))
	sm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes*4], -0.1, 0.1))
	sb = tf.Variable(tf.zeros([1, num_nodes*4])) 
	# Variables saving state across unrollings.
	saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
	saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
	# Classifier weights and biases.
	w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
	b = tf.Variable(tf.zeros([vocabulary_size]))

	def lstm_cell(i, o, state):
		smatmul = tf.matmul(i, sx) + tf.matmul(o, sm) + sb
		smatmul_input, smatmul_forget, update, smatmul_output = tf.split(1, 4, smatmul)
		input_gate = tf.sigmoid(smatmul_input)
		forget_gate = tf.sigmoid(smatmul_forget)
		output_gate = tf.sigmoid(smatmul_output)
		state = forget_gate * state + input_gate * tf.tanh(update)
		return output_gate * tf.tanh(state), state

	# Input data.
	train_data = list()
	for _ in range(num_unrollings + 1):
		train_data.append(tf.placeholder(tf.float32, shape=[batch_size,vocabulary_size]))
	train_inputs = train_data[:num_unrollings]
	train_labels = train_data[1:]  # labels are inputs shifted by one time step.
	# Unrolled LSTM loop.
	outputs = list()
	output = saved_output
	state = saved_state
	for i in train_inputs:
		i_embed = tf.nn.embedding_lookup(vocabulary_embeddings, tf.argmax(i, dimension=1))
		output, state = lstm_cell(i_embed, output, state)
		outputs.append(output)
	
    ####################################################################################################
	# Loss.
	with tf.control_dependencies([saved_output.assign(output),saved_state.assign(state)]):
		logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf.concat(0, train_labels)))
	# Learning Rate
	global_step = tf.Variable(0)
	learning_rate = tf.train.exponential_decay(10.0, global_step, 5000, 0.1, staircase=True)
	# Optimizer
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	gradients, v = zip(*optimizer.compute_gradients(loss))
	gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
	optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)
	# Predictions.
	train_prediction = tf.nn.softmax(logits)
	#####################################################################################################
	
	# Sampling and validation eval: batch 1, no unrolling.
	sample_input = tf.placeholder(tf.float32, shape=[1, vocabulary_size])
	sample_input_embedding = tf.nn.embedding_lookup(vocabulary_embeddings, tf.argmax(sample_input, dimension=1))
	saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
	saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
	reset_sample_state = tf.group(saved_sample_output.assign(tf.zeros([1, num_nodes])),saved_sample_state.assign(tf.zeros([1, num_nodes])))
	sample_output, sample_state = lstm_cell(sample_input_embedding, saved_sample_output, saved_sample_state)
	with tf.control_dependencies([saved_sample_output.assign(sample_output),saved_sample_state.assign(sample_state)]):
		sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))

	init = tf.initialize_all_variables()
	saver = tf.train.Saver()

num_steps = 7001
summary_frequency = 100

with tf.Session(graph=graph) as session:
	session.run(init)
	print('Initialized')
	mean_loss = 0
	for step in range(num_steps):
		batches = train_batches.next()
		feed_dict = dict()
		for i in range(num_unrollings + 1):
			feed_dict[train_data[i]] = batches[i]
		_, l, predictions, lr = session.run([optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
		mean_loss += l
		if step % summary_frequency == 0:
			valid_logprob = 0
			for _ in range(valid_size):
				b = valid_batches.next()
				predictions = sample_prediction.eval({sample_input: b[0]})
				valid_logprob = valid_logprob + logprob(predictions, b[1])
			print('Validation set perplexity: %.2f' % float(np.exp(valid_logprob / valid_size)))
	save_path = saver.save(session, "/home/lohani/Desktop/latest/model.model")
	print("Model saved in file: %s" % save_path)
	
# Running a new session
print("Starting 2nd session...")
with tf.Session(graph=graph) as session:
	# Initialize variables
	session.run(init)
	# Restore model weights from previously saved model
	saver.restore(session, "/home/lohani/Desktop/latest/model.model")
	# print("Model restored from file: %s" % save_path)
	for _ in range(5):
		feed = sample(random_distribution())
		sentence = words(feed)[0]
		print(words(feed)[0])
		reset_sample_state.run()
		for _ in range(79):
			prediction = sample_prediction.eval({sample_input: feed})
			feed = sample(prediction)
			sentence = sentence + ' ' + words(feed)[0]
		print(sentence)
	print('=' * 80)


