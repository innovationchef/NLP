import os
import sys
import copy
import numpy as np
from numpy import dot
from numpy.linalg import norm
from porter_stemmer import PorterStemmer

class VectorSpace:
	"""     	doc1	doc2	doc3	doc4	....    docN
		term1	5		8		9		0				5	
		term2	2		6		7		3				1			
		term3	7		2		4		9				7
		term4	4		2		0		0				6
		.....
		termM	0		7		4		0				9

	This is the vector space I am talking about.
	The document vectors are --> vec(doc1) = 5*term1 + 2*term2 + 7*term3 + ..... + 0*termM
	If a term occurs in the document, then the value in the vector is non-zero.
	"""
	vector_space = []
	allwords_to_index_dict = []
	global stopwords 
	global unique_vocabulary_list
	unique_vocabulary_list = []
	stopwords = ['a', 'the']

	def __init__(self, documents = []):
		self.vector_space = []
		if len(documents) > 0:
			self.build_vec_space(documents)

	def build_vec_space(self, documents):
		"""A vector space is basically a set of document vectors stacked vertically"""
		"""Documents are a list of strings that is passed here."""

		"""To make a VSpace, we need list of all possible words that occur in all docs"""
		self.allwords_to_index_dict  = self.get_dict_of_unique_words(documents)

		"""for vec(doc) We need to create a null vector and put the frequencies corresponding to each term"""
		matrix = [self.make_vector(document) for document in documents]
		"""
		Now we have this -->
			vec(doc1) = 5*term1 + 8*term2 + 9*term3 + ..... + 5*termM
			vec(doc2) = 2*term1 + 6*term2 + 7*term3 + ..... + 1*termM
			vec(doc3) = 7*term1 + 2*term2 + 4*term3 + ..... + 7*termM
		"""
		# print np.shape(matrix)
		self.vector_space = matrix

	def get_dict_of_unique_words(self, document_list):
		""" create the keyword associated to the position of the elements within the document vectors """
		global unique_vocabulary_list
		vocabulary_list = self.tokenise_and_remove_stop_words(document_list)
		unique_vocabulary_list = self.remove_duplicates(vocabulary_list)
		vector_index={}
		offset=0
		#Associate a position with the keywords which maps to the dimension on the vector used to represent this word
		for word in unique_vocabulary_list:
			vector_index[word] = offset
			offset += 1
		return vector_index  #(keyword:position)

	def make_vector(self, word_string):
		"""First create null vectors of form --> vec(doc1) = 0*term1 + 0*term2 + 0*term3 + ..... + 0*termM"""
		vector = [0] * len(self.allwords_to_index_dict)

		word_list = self.tokenise_and_remove_stop_words(word_string.split(" "))

		for word in word_list:
			vector[self.allwords_to_index_dict[word]] += 1; #Use simple Term Count Model
		"""Now we have this vec(doc1) = 5*term1 + 2*term2 + 7*term3 + ..... + 0*termM"""
		return vector

	def tokenise_and_remove_stop_words(self, document_list):
		if not document_list:
		  return []
		  
		vocabulary_string = " ".join(document_list)
				
		tokenised_vocabulary_list = self.tokenise(vocabulary_string)
		clean_word_list = self.remove_stop_words(tokenised_vocabulary_list)
		return clean_word_list

	def remove_stop_words(self, list):
		""" Remove common words which have no search value """
		return [word for word in list if word not in stopwords ]

	def tokenise(self, string):
		""" break string up into tokens and stem words """
		stemmer = PorterStemmer()
		string = self.clean(string)
		words = string.split(" ")
		return [stemmer.stem(word, 0, len(word)-1) for word in words]

	def clean(self, string):
		""" remove any nasty grammar tokens from string """
		string = string.replace(".","")
		string = string.replace("\s+"," ")
		string = string.lower()
		return string

	def relatedness(self, document_id):
		""" find documents that are related to the document indexed by passed Id within the document Vectors"""
		ratings = [self.cosine(self.vector_space[document_id], document_vector) for document_vector in self.vector_space]
		# ratings.sort(reverse = True)
		return ratings

	def search(self, searchList):
		""" search for documents that match based on a list of terms """
		queryVector = self.build_query_vector(searchList)

		ratings = [self.cosine(queryVector, documentVector) for documentVector in self.vector_space]
		# ratings.sort(reverse=True)
		return ratings

	def build_query_vector(self, term_list):
		""" convert query string into a term vector """
		anylist = []
		for word in term_list:
			if not word in unique_vocabulary_list:
				term_list.remove(word)
			else: dummy=1
		query = self.make_vector(" ".join(term_list))
		return query

	def remove_duplicates(self, list):
		""" remove duplicates from a list """
		return set((item for item in list))
		
	def cosine(self, vector1, vector2):
		""" related documents j and q are in the concept space by comparing the vectors :
			cosine  = ( V1 * V2 ) / ||V1|| x ||V2|| """
		return float(dot(vector1,vector2) / (norm(vector1) * norm(vector2)))


def create_docString(i):
	"""i = number of file you want to process"""
	fname = "cancer_data/" + str(i) + ".txt"
	return fname


doc = []
for i in range(0,100):
	with open(create_docString(i), 'r') as the_file:
		data = the_file.read()
	doc.append(data)
MyVSpace = VectorSpace(doc)
# print MyVSpace.search(['chu', 'cancer', 'virtual'])