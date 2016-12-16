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
	def __init__(self, current_input=[], prev_output=[], prev_cell=[]):
		self.x_t = current_input
		self.h_t_prev = prev_output
		self.c_t_prev = prev_cell
		batch_size, hidden = np.shape(prev_output)
		batch_size, vec_size = np.shape(current_input)
		self.n_hidden = hidden
		self.n_batch = batch_size
		self.n_vector = vec_size
		self.h_t_out = np.empty((self.n_batch, self.n_hidden))
		self.c_t_present = np.empty((self.n_batch, self.n_hidden))
		self.y_t_hat = np.empty((self.n_batch, self.n_vector))
		self.h_t()

	def input_gate(self):
		x_t, h_t_prev = self.x_t, self.h_t_prev
		W_i = np.random.uniform(-np.sqrt(1./self.n_vector), np.sqrt(1./self.n_vector), (self.n_vector, self.n_hidden))
		U_i = np.random.uniform(-np.sqrt(1./self.n_hidden), np.sqrt(1./self.n_hidden), (self.n_hidden, self.n_hidden))
		b_i = np.zeros((self.n_hidden,1))
		calculation = mat_add(mat_mul(x_t,W_i), mat_mul(h_t_prev,U_i))
		activation = sigmoid(calculation)
		return activation
	def output_gate(self):
		x_t, h_t_prev = self.x_t, self.h_t_prev
		W_o = np.random.uniform(-np.sqrt(1./self.n_vector), np.sqrt(1./self.n_vector), (self.n_vector, self.n_hidden))
		U_o = np.random.uniform(-np.sqrt(1./self.n_hidden), np.sqrt(1./self.n_hidden), (self.n_hidden, self.n_hidden))
		b_o = np.zeros((self.n_hidden,1))
		calculation = mat_add(mat_mul(x_t, W_o), mat_mul(h_t_prev, U_o))
		activation = sigmoid(calculation)
		return activation
	def forget_gate(self):
		x_t, h_t_prev = self.x_t, self.h_t_prev
		W_f = np.random.uniform(-np.sqrt(1./self.n_vector), np.sqrt(1./self.n_vector), (self.n_vector, self.n_hidden))
		U_f = np.random.uniform(-np.sqrt(1./self.n_hidden), np.sqrt(1./self.n_hidden), (self.n_hidden, self.n_hidden))
		b_f = np.zeros((self.n_hidden,1))
		calculation = mat_add(mat_mul(x_t,W_f), mat_mul(h_t_prev,U_f))
		activation = sigmoid(calculation)
		return activation
	def c_present_tilde(self):
		x_t, h_t_prev = self.x_t, self.h_t_prev
		W_c = np.random.uniform(-np.sqrt(1./self.n_vector), np.sqrt(1./self.n_vector), (self.n_vector, self.n_hidden))
		U_c = np.random.uniform(-np.sqrt(1./self.n_hidden), np.sqrt(1./self.n_hidden), (self.n_hidden, self.n_hidden))
		b_c = np.zeros((self.n_hidden,1))
		calculation = mat_add(mat_mul(x_t,W_c), mat_mul(h_t_prev,U_c))
		activation = sigmoid(calculation)
		return activation
	def c_present(self):
		c_t_prev = self.c_t_prev
		i = self.input_gate()
		f = self.forget_gate()
		c_pres_t = self.c_present_tilde()
		c_present = mat_add(hadamard_prod(i, c_pres_t), hadamard_prod(f, c_t_prev))
		activation = tanh(c_present)
		return activation
	def h_t(self):
		o = self.output_gate()
		activated_c_present = self.c_present()
		calculation = hadamard_prod(o, activated_c_present)
		self.h_t_out = calculation
		return calculation
	def y_output(self):
		V = np.random.uniform(-np.sqrt(1./self.n_hidden), np.sqrt(1./self.n_hidden), (self.n_hidden, self.n_vector))
		h_t = self.h_t()
		calculation = mat_mul(h_t,V)
		activation = softmax(calculation)
		self.y_t_hat = activation
		return activation
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
				B[row][col] = math.tanh(A[row][col])
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
	def __init__(self, batches, batch_size, n_hidden):
		self.batches = batches
		self.n_hidden = n_hidden

		self.word_batch = {}
		self.h_o = np.random.uniform(-np.sqrt(1./self.n_hidden), np.sqrt(1./self.n_hidden), (batch_size, self.n_hidden))
		self.c_t_prev = np.random.uniform(0, 1, (batch_size, self.n_hidden))
		self.output_batches = []
		self.word_batch = {}
		h_t_prev = self.h_o
		c_t_prev = self.c_t_prev
		for batch in self.batches[:-1]:
			MU = memory_unit(current_input=batch, prev_output=h_t_prev, prev_cell=c_t_prev)
			self.word_batch[1] = MU
			h_t_prev = MU.h_t_out
			c_t_prev = MU.c_present()
			self.output_batches.append(MU.h_t())
		print h_t_prev

						
#http://www.thushv.com/sequential_modelling/long-short-term-memory-lstm-networks-implementing-with-tensorflow-part-2/

# MU = memory_unit(current_input=vocab_df['walking'], prev_output=[], prev_cell=[])
batches = list()
batch_size = 10
for i in range(25):
	a = np.random.uniform(0, 1, (batch_size, 27))
	batches.append(a)

neural_network(batches,batch_size,10)