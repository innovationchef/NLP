"""
Description of Data
I have 1800 queries and for each query I have multiple related documents
say, Q1 has 10 D+, Q2 has 15 D+ and Q3 has 2 D+ and so on.
I am assuming that the for Q1, the documents of Q2 and Q3 and unrelated and hence unclicked.

Problem Statement -- 
Document retrieval for a given query and multiple documents.
Input = Query (say Q1)
Output = List of Documents (10 D+ and rest Documents followed by this in decreasing probability)
"""


from itertools import groupby, cycle, tee, islice, chain, izip
import csv
import re
import random
import itertools
import numpy as np
import tensorflow as tf
from numpy import random
from collections import defaultdict, OrderedDict, Counter

def clean_str(string):
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", " n\'t", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)
	return string.strip().lower()
def load_data_and_labels():
	columns = defaultdict(list)
	with open("new_data.csv", 'rb') as f:
		reader = csv.reader(f)
		for row in reader:
			for (i,v) in enumerate(row):
				columns[i].append(v)
	queries = []
	docs = columns[1]
	hehe = []
	j = -1
	cursor = []
	labels = np.zeros((1749, len(docs)))
	for i, q in enumerate(columns[0]):
		if not q == '':
			cursor.append(i)
			j+=1
			queries.append(q)
	cursor.append(len(columns[1]))
	return docs, queries, cursor
def pad_sentences(sentences, padding_word=" </s>"):
	sen = []
	for i in range(len(sentences)):
		sentence = clean_str(sentences[i])
		sen.append(sentence.split(' '))
	for i in range(len(sen)):
		sen[i].insert(0, "</s> ")
		sen[i].insert(len(sen[i]), " </s>")
	return sen
def create_dict_trigram(docs, more_docs):
	letter_gram_size = 3
	ngrams = []
	for doc in docs:
		for word in doc:
			word="#"+word+"#"
			for i in range(len(word)-letter_gram_size+1):
				ngrams.append(word[i:i+letter_gram_size])
	for _doc in more_docs:
		for _word in _doc:
			_word="#"+_word+"#"
			for _i in range(len(_word)-letter_gram_size+1):
				ngrams.append(_word[_i:_i+letter_gram_size])
	DICT = list(set(ngrams))     
	return DICT
def create_embeddings(data, DICT):
	letter_gram_size = 3
	j = 1
	batch = []
	while j < (len(data)-1):
		wordWindow = []
		wordWindow.append("#"+data[j-1]+"#")
		wordWindow.append("#"+data[j]+"#")
		wordWindow.append("#"+data[j+1]+"#")
		embedding = []
		for word in wordWindow:	
			word_breakup =[]
			rep = np.zeros((1, len(DICT)))
			for i in range(len(word)-letter_gram_size+1):
				word_breakup.append(word[i:i+letter_gram_size])
			for trigram in word_breakup:
				rep[0][DICT.index(trigram)] = 1
			embedding.append(rep[0])
		batch.append(embedding)
		j += 1
	return np.array(batch)

print "Preparing Data"
docs, queries, cursor = load_data_and_labels()
docs_padded = pad_sentences(docs)
queries_padded = pad_sentences(queries)
trigram_dict = create_dict_trigram(queries_padded, docs_padded) 
print "Data Prepared"

#########################################################################################################
"""
**** What happened till here?

I have 1800 queries and x number for D+ documents associated with each query (x varies for each query)
I have a dictionary of trigrams created from the query and documents in training examples. The size of the dict will be the 
embedding size of the word.

Query length is variable and so is document size. So, we create a representation matrix for all queries and documents as follows - -
Eg : How to learn Deap Learning?

n_rows = query length
n_cols = dict_size

</s>		0 0 0 0 .......... 0 0 0 0 0 0 0
How			1 0 0 0 .......... 0 0 0 0 1 0 0
to 			0 0 0 0 .......... 0 0 0 0 1 0 1
learn 		1 0 1 0 .......... 0 0 1 0 0 0 0
deep 		1 0 0 1 .......... 0 0 0 0 1 0 0
learning 	0 0 0 0 .......... 0 1 0 0 0 0 1
</s>		0 0 0 0 .......... 0 0 0 0 0 0 0

**** What will be fed to the TF Graph?

for query in queries:
	for doc in queries #remember, we have multiple docs associated with each query
		Feed == query + doc + 3 unrelated/unclicked docs from any other random question
"""
#########################################################################################################

print "Initializing Parameters"
n_window = 3
n_channel = 1 # it is one for Text CNN
n_embedding = len(trigram_dict) # Described Above
n_maxPool = 300	# as in Paper
n_semanticSpace = 128	# as in paper
n_docs = 3 # number of papers we are taking for each pos_doc
steps = 3 # in each step we are feeding 1 Query and all related docs, for in 1200 steps we will feed 1200 queries and their related docs
learning_rate = 0.001 # kept low and fixed
print "Parameters Initialized"
print "Creating Tensorflow Graph"

#########################################################################################################
"""
tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
Computes a 2-D convolution given 4-D input and filter tensors.
** value: A 4D Tensor. Must be of type float32 or float64 |||||| [batch, in_height, in_width, in_channels]
** filters: A 4D Tensor. Must have the same type as input |||||| [filter_height, filter_width, in_channels, out_channels]
batch = len of sentence which varies (None)
in_height = window size (CNN goes over this window (</s> How to) (How to learn) (to learn deep) (learn deep learning?) (deep learning? </s>))
in_width = embedding size
in_channels = 1

filter_height = window_size
filter_width = embedding size
in_channels = 1 (for text CNN)
out_channels = n_maxPool
"""
#########################################################################################################


graph = tf.Graph()
with graph.as_default():
	"""QUERY"""
	Q_input_shape = [None, n_window, n_embedding, n_channel]
	query = tf.placeholder(tf.float32, Q_input_shape, name="query")
	filter_shape = [n_window, n_embedding, 1, n_maxPool]
	W1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W1")
	b1 = tf.Variable(tf.constant(0.1, shape=[n_maxPool]), name="b1")
	Q_conv = tf.nn.conv2d(query, W1, strides=[1, 1, 1, 1], padding="VALID", name="Q_conv")
	Q_h = tf.nn.tanh(tf.nn.bias_add(Q_conv, b1), name="tanh")
	Q_pooled = tf.nn.max_pool(Q_h, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1],padding='VALID',name="pool")
	Q_h_pool_flat = tf.squeeze(Q_pooled)
	semantic_filter_shape = [n_maxPool, n_semanticSpace]
	W2 = tf.Variable(tf.truncated_normal(semantic_filter_shape, stddev=0.1), name="W2")
	b2 = tf.Variable(tf.constant(0.1, shape=[n_semanticSpace]), name="b2")	
	query_sem = tf.nn.xw_plus_b(Q_h_pool_flat, W2, b2)

	# size of query_sem = [batch_size, 128]

	"""METHOD"""
	def doc_conv(doc, variables_dict):
		D_conv = tf.nn.conv2d(doc, variables_dict["W3"], strides=[1, 1, 1, 1], padding="VALID", name="D_conv")
		D_h = tf.nn.tanh(tf.nn.bias_add(D_conv, variables_dict["b3"]), name="tanh")
		return D_h
	def doc_max(D_h):	
		D_pooled = tf.nn.max_pool(D_h, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1],padding='VALID',name="pool")
		D_h_pool_flat = tf.squeeze(D_pooled)
		return D_h_pool_flat
	def docs_sem(D_h_pool_flat, variables_dict):
		semantic_filter_shape = [n_maxPool, n_semanticSpace]
		doc_sem = tf.nn.xw_plus_b(D_h_pool_flat, variables_dict["W4"], variables_dict["b4"])	
		return doc_sem

	# size of doc_sem = [batch_size, 128]

	variables_dict = {
	"W3": tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W3"),
	"b3": tf.Variable(tf.constant(0.1, shape=[n_maxPool]), name="b3"),
	"W4": tf.Variable(tf.truncated_normal(semantic_filter_shape, stddev=0.1), name="W4"),
	"b4": tf.Variable(tf.constant(0.1, shape=[n_semanticSpace]), name="b4")	
	}

	"""POS DOCUMENTS"""
	pos_packed_doc = tf.placeholder(tf.float32, shape=[None, n_window, n_embedding, n_channel], name="pos_packed_doc")
	pos_conv = doc_conv(pos_packed_doc, variables_dict)
	pos_max_pool = doc_max(pos_conv)
	pos_doc_sem = docs_sem(pos_max_pool, variables_dict)
	
	"""NEG DOCUMENTS"""
	neg_packed_docs = tf.placeholder(tf.float32, shape=[None, n_window, n_embedding, n_channel], name="neg_packed_docs")
	_positions = tf.placeholder(tf.float32, shape=[n_docs,1], name="positions")
	point = tf.unpack(_positions)
	
	new = []
	new.append(tf.slice(neg_packed_docs, [0,0,0,0], [tf.cast(tf.reduce_sum(point[0]), dtype=tf.int32),n_window,n_embedding,1]))
	for i in range(n_docs-1):
		new.append(tf.slice(neg_packed_docs, [tf.cast(tf.reduce_sum(point[i]), dtype=tf.int32)-1,0,0,0], [tf.cast(tf.reduce_sum(point[i+1]), dtype=tf.int32),n_window,n_embedding,1]))

	neg_conv = [doc_conv(new[i], variables_dict)  for i in range(n_docs)]
	neg_max_pool = [doc_max(neg_conv[i]) for i in range(n_docs)]
	neg_docs_sem = [docs_sem(neg_max_pool[i], variables_dict)  for i in range(n_docs)]


	"""TRAINING CALCULATIONS"""
	def cosine(Q,D):
		normed_x = tf.nn.l2_normalize(Q, dim=1)
		normed_y = tf.nn.l2_normalize(D, dim=1)
		calculate = tf.matmul(normed_x, tf.transpose(normed_y, [1, 0]))
		return calculate
	
	# PLEASE NOTE HERE THAT SIZE OF MY COSINE SIMILARITY MATRIX IS (batch_size of query, batch_size of doc) = [B1, B2]
	#  Michael, in your previous code, it was different and it is being fixed by you in the lambda layer
	# I not very sure about the size of matrix in your new code  R_Q_D_p = merge([query_sem, pos_doc_sem], mode = "cos")

	pos_logits = tf.reduce_sum(cosine(query_sem, pos_doc_sem)) # 3.02
	neg_logits = [tf.reduce_sum(cosine(query_sem, neg_docs_sem[i])) for i in range(n_docs)]		# [1.01 0.04 0.35]  since, n_docs = 3

	# Here in logits, I have simply converted the cosine matrix of size [B1, B2] to a single number [1] 

	neg_exp = [tf.exp(neg_logits[i]) for i in range(n_docs)]
	pos_exp = tf.exp(pos_logits)
	neg_summation = tf.add_n(neg_exp)
	total_summation = tf.add(neg_summation, pos_exp)
	prob_Dplus_given_Q = tf.truediv(pos_exp, total_summation)	
	
	# prob_Dplus_given_Q give the softmax without the gamma term included in it

	loss = -tf.log(prob_Dplus_given_Q)

	# here is where I am stuck!! I have a query and a pos_doc and 3 neg_docs in this graph in each step. How am I supposed to find the loss ?
	
	'''training'''
	global_step = tf.Variable(0, name="global_step", trainable=False)
	decay_step = 1
	decay_rate = 0.9
	learning_r = tf.train.exponential_decay(learning_rate, global_step, decay_step, decay_rate)
	opt = tf.train.AdamOptimizer(learning_r)
	train_step = opt.minimize(loss, global_step=global_step)

	# """RESULTS"""
	init = tf.initialize_all_variables()
	saver = tf.train.Saver()


print "Tensorflow Graph Created"
print "starting training session"
with tf.Session(graph=graph) as session:
	session.run(init)
	print('Initialized')
	nn = 0
	for iquer in range(steps):
		for idoc in range(cursor[iquer+1]-cursor[iquer]):
			position = np.zeros([n_docs,1]).astype(int)
			query_matrix = np.expand_dims(create_embeddings(queries_padded[iquer], trigram_dict), axis = -1)
			pos_doc_matrix = np.expand_dims(create_embeddings(docs_padded[cursor[iquer]+idoc], trigram_dict), axis = -1) 
			neg_doc_matrix = np.expand_dims(create_embeddings(docs_padded[random.choice(range(0,cursor[iquer]) + range(cursor[iquer+1],cursor[steps]))], trigram_dict), axis = -1)
			position[0][0] = np.shape(neg_doc_matrix)[0]
			i=1
			while i < n_docs:
				n_doc_matrix = np.expand_dims(create_embeddings(docs_padded[random.choice(range(0,cursor[iquer]) + range(cursor[iquer+1],cursor[steps]))], trigram_dict), axis = -1)
				position[i][0] = np.shape(n_doc_matrix)[0]
				neg_doc_matrix = np.concatenate((neg_doc_matrix, n_doc_matrix), axis=0)
				i += 1
			feed_dict = {query: query_matrix, pos_packed_doc: pos_doc_matrix, neg_packed_docs: neg_doc_matrix, _positions: position}
			lost, _ = session.run([loss, train_step], feed_dict=feed_dict)
			print "Epoch==> ", iquer+1, " ==LOSS==>  ", lost
	
		if iquer % 100 is 0:
			save_path = saver.save(session, "./model" + str(nn) + ".ckpt", global_step=steps, write_meta_graph=False)
			print "iquer --> ", iquer, "  model number -->  ", nn 
			print("Model saved in file: %s" % save_path)
			print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
			nn += 1
 
