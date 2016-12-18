import math
import numpy as np
# from embedding.word_vector import word_embedding
# vocab_df = word_embedding("gandhi.txt")
"""
nodes/n_hidden = 64
word_dimention = (1*27)
batch_size = 10

LSTM Cell ::::
Input to Cell
**************current_input, prev_output, prev_cell*****************
current_input = 10*27
prev_output = 10*64
prev_cell = 10*64

Cell Internal Processing
**************Input gate, Forget gate, c_present_tilda, c_present, Output gate***************
IG = 10*64
FG = 10*64
c_present_tilda = 10*64
c_present = 10*64
OG = 10*64
Left Side Wt Matrices = 27*64
Right Side Weight Matrices = 64*64
Biases = 64*1
final outtput to next layer = 10*64
"""
class memory_unit():
	global mat_mul
	global mat_add
	global mat_diff 
	global sigmoid
	global softmax 
	global tanh
	global hadamard_prod
	global error_mat
	def __init__(self, current_input=[],next_input=[], prev_output=[], prev_cell=[]):
		
		self.x_t = current_input
		self.h_t_prev = prev_output
		self.c_t_prev = prev_cell
		batch_size, hidden = np.shape(prev_output)
		batch_size, vec_size = np.shape(current_input)
		self.n_hidden = hidden
		self.n_batch = batch_size
		self.n_vector = vec_size
		self.W = np.random.uniform(-np.sqrt(1./self.n_vector), np.sqrt(1./self.n_vector), (4, self.n_vector, self.n_hidden))
		self.U = np.random.uniform(-np.sqrt(1./self.n_hidden), np.sqrt(1./self.n_hidden), (4, self.n_hidden, self.n_hidden))
		self.b = np.zeros((self.n_hidden,1))
		self.h_t_out = np.empty((self.n_batch, self.n_hidden))
		self.c_t_present = np.empty((self.n_batch, self.n_hidden))
		self.y_t_hat = np.empty((self.n_batch, self.n_vector))
		self.y_t = next_input
		self.delta_h_t = np.empty((self.n_batch, self.n_vector))
		self.delta_o_t = np.empty((self.n_batch, self.n_vector))
		self._build()

	def _build(self):
		self.c_present()
		self.h_t()
		self.y_output()
		self.calc_delta_h_t()
		# self.calc_delta_o_t()
	def input_gate(self):
		x_t, h_t_prev= self.x_t, self.h_t_prev
		calculation = mat_add(mat_mul(x_t,self.W[0]), mat_mul(h_t_prev,self.U[0]))
		activation = sigmoid(calculation)
		return activation
	def output_gate(self):
		x_t, h_t_prev = self.x_t, self.h_t_prev
		calculation = mat_add(mat_mul(x_t, self.W[1]), mat_mul(h_t_prev, self.U[1]))
		activation = sigmoid(calculation)
		return activation
	def forget_gate(self):
		x_t, h_t_prev = self.x_t, self.h_t_prev
		calculation = mat_add(mat_mul(x_t,self.W[2]), mat_mul(h_t_prev,self.U[2]))
		activation = sigmoid(calculation)
		return activation
	def c_present_tilde(self):
		x_t, h_t_prev = self.x_t, self.h_t_prev
		calculation = mat_add(mat_mul(x_t,self.W[3]), mat_mul(h_t_prev,self.U[3]))
		activation = sigmoid(calculation)
		return activation
	def c_present(self):
		c_t_prev = self.c_t_prev
		i = self.input_gate()
		f = self.forget_gate()
		c_pres_t = self.c_present_tilde()
		c_present = mat_add(hadamard_prod(i, c_pres_t), hadamard_prod(f, c_t_prev))
		self.c_t_present = tanh(c_present)
	def h_t(self):
		o = self.output_gate()
		activated_c_present = self.c_t_present
		calculation = hadamard_prod(o, activated_c_present)
		self.h_t_out = calculation
	def y_output(self):
		V = np.random.uniform(-np.sqrt(1./self.n_hidden), np.sqrt(1./self.n_hidden), (self.n_hidden, self.n_vector))
		h_t = self.h_t_out
		calculation = mat_mul(h_t,V)
		activation = softmax(calculation)
		self.y_t_hat = activation
	def calc_delta_h_t(self):
		self.delta_h_t = error_mat(self.y_t,self.y_t_hat) 
	def calc_delta_o_t(self):
		self.delta_o_t = hadamard_prod(self.delta_h_t, tanh(self.c_t_present))
	def error_mat(y_actual,y_calculated):
		rows, cols = np.shape(y_actual)
		error = np.empty(np.shape(y_calculated))
		for row in range(rows):
			for col in range(cols):
				error[row][col] =  y_actual[row][col]*math.log(y_calculated[row][col]) + (1-y_actual[row][col])*math.log(1-y_calculated[row][col])
		return error
	def sigmoid(A):
		rows, cols = np.shape(A)
		B = np.empty(np.shape(A))
		for row in range(rows):
			for col in range(cols):
				B[row][col] = 1 / (1 + math.exp(-A[row][col]))
		return B
	def tanh(A):
		rows, cols = np.shape(A)
		B = np.empty(np.shape(A))
		for row in range(rows):
			for col in range(cols):
				B[row][col] = math.tanh(A[row][col])
		return B
	def softmax(A):
		rows, cols = np.shape(A)
		B = np.empty(np.shape(A))
		for row in range(rows):
			for col in range(cols):
				B[row][:] = np.exp(A[row][:]) / np.sum(np.exp(A[row][:]))
		return B
	def hadamard_prod(A,B):
		C = np.multiply(A,B)
		return C
	def mat_mul(A=[],B=[]):
		product = np.dot(A,B)
		return product
	def mat_add(A, *argv):
		for arg in argv:
			A = np.add(A, arg)
		return A
	def mat_diff(A,B):
		calculation = self.mat_add(A,-B)
		return calculation
	
class neural_network(object):
	def __init__(self, wordvec, n_words, n_hidden):
		self.wordvec = wordvec
		self.n_hidden = n_hidden
		self.n_words = n_words
		self.word_batch = {}
		self.h_o = np.random.uniform(-np.sqrt(1./self.n_hidden), np.sqrt(1./self.n_hidden), (1, self.n_hidden))
		self.c_t_prev = np.random.uniform(0, 1, (1, self.n_hidden))
		self.output_wordvec = []
		self.word_batch = {}
		h_t_prev = self.h_o
		c_t_prev = self.c_t_prev
		i = 0
		for word in self.wordvec[:-1]:
			next_word = self.wordvec[i+1]
			MU = memory_unit(current_input=word,next_input =next_word, prev_output=h_t_prev, prev_cell=c_t_prev)
			self.word_batch[1] = MU
			h_t_prev = MU.h_t_out
			c_t_prev = MU.c_t_present
			delta_h_t = MU.delta_h_t
			self.output_wordvec.append(MU.h_t())
			i+=1
		print next_word, MU.y_t_hat
						
# http://www.thushv.com/sequential_modelling/long-short-term-memory-lstm-networks-implementing-with-tensorflow-part-2/
# http://arunmallya.github.io/writeups/nn/lstm/index.html#/6
# http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/
# MU = memory_unit(current_input=vocab_df['walking'], prev_output=[], prev_cell=[])
wordvec = list()
n_words = 30
for i in range(n_words):
	wordvec.append(np.random.uniform(0, 1, (1, 27)))  	

neural_network(wordvec,n_words,10)