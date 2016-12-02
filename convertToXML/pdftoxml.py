########################################################################
#Step 1 : Take a file in PDF Format
#Step 2 : Convert it to txt file with html tags
#Step 3 : Use this txt file to generate a xml file
########################################################################


#####################################################################

import subprocess
import numpy as np
from numpy  import array
from collections import Counter
from HTMLParser import HTMLParser	
from xml.etree.ElementTree import Element, SubElement, tostring

import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

import re
import os
rxcountpages = re.compile(r"/Type\s*/Page([^s]|$)", re.MULTILINE|re.DOTALL)
def count_pages(filename):
    data = file(filename,"rb").read()
    return len(rxcountpages.findall(data))

# create a subclass and override the handler methods
class Stack:
		def __init__(self):
			self.container = []  # You don't want to assign [] to self - when you do that, you're just assigning to a new local variable called `self`.  You want your stack to *have* a list, not *be* a list.

		def isEmpty(self):
			return len(self.container) == 0   # While there's nothing wrong with self.container == [], there is a builtin function for that purpose, so we may as well use it.  And while we're at it, it's often nice to use your own internal functions, so behavior is more consistent.

		def push(self, item):
			self.container.append(item)  # appending to the *container*, not the instance itself.

		def pop(self):
			try : 
				return self.container.pop()  # pop from the container, this was fixed from the old version which was wrong
			except IndexError:
				dummy = 1

		def peep(self):
			temp = self.pop()
			self.push(temp)
			return temp

		def size(self):
			return len(self.container)  # length of the container

class OverallParser(HTMLParser):
	
	def handle_starttag(self, tag, attrs):
		global currentPage
		global TagAttr
		if tag == str("br"): dummy = 1
		else : 
			pages[currentPage].push(tag)
			if TagAttr.get(tag) is not None:
				TagAttr[tag].append(attrs)

	def handle_endtag(self, tag):
		global currentPage
		if pages[currentPage].peep() == tag : pages[currentPage].pop()
		else: 
			pages[currentPage].push(tag)

	def handle_data(self, data):
		global currentPage
		data = data.split()
		if set(['Page',str(currentPage+1)]) == set(data): currentPage += 1
		if len(data) is 0: dummy = 1
		else : pages[currentPage].push(data)

class UniqueParser(HTMLParser):
	
	def handle_starttag(self, tag, attrs):
		global currentPage2
		global unique
		if tag == str("br"): dummy = 1
		else : 
			element = array([element[1] for element in attrs])
			pages2[currentPage2].push(tag)
			if set(element).issubset(set(unique)):
				pages2[currentPage2].push(str(element).replace('[\'','').replace('\']',''))

	def handle_endtag(self, tag):
		global currentPage2
		if pages2[currentPage2].peep() == tag : pages2[currentPage2].pop()
		else: 
			pages2[currentPage2].push(tag)

	def handle_data(self, data):
		global currentPage2
		data = data.split()
		if set(['Page',str(currentPage2+1)]) == set(data): currentPage2 += 1
		if len(data) is 0: dummy = 1
		else : pages2[currentPage2].push(data)
		
def reverse(stack):
	new_stack = Stack()
	while not stack.isEmpty():
		new_stack.push(stack.pop())
	return new_stack
			
def readFile(fname):
	file = fname
	with open(file, 'r') as myfile:
		data=myfile.read().replace('\n', '')
	return data
			
def uniqueAttrs(unique, counts):
	delete = []
	for i in range(0,len(counts)):
		if counts[i]<5:
			delete.insert(0, i)
	for items in delete:
		unique=np.delete(unique, items)
		counts=np.delete(counts, items)
	unique = np.delete(unique, len(unique)-1)
	counts = np.delete(counts, len(counts)-1)
	return unique, counts

file_name = str(sys.argv[1])
TotalPages = count_pages(file_name)
# print TotalPages, file_name
subprocess.call(['./converttotxt.sh'+' '+file_name], shell = True)
output_file = file_name[:-4]
output_file = output_file+'.txt'
xml_file = file_name[:-4]+'.xml'
data = readFile(output_file)

TagAttr = {}
TagAttr['span'] = []
TagAttr['div'] = []

totalPages = TotalPages+2

currentPage = 0
pages = {}
for i in range(0, totalPages):
	pages[i] = i
for key, val in pages.items():
	pages[key] = Stack()
read = 0
parser = OverallParser()
parser.feed(data)
for key, val in pages.items():
	pages[key] = reverse(pages[key])

unique = []
mylist = array(TagAttr['span'])
unique, counts = np.unique(mylist, return_counts=True)
unique, counts = uniqueAttrs( unique, counts)
maxAttr = np.where(counts==counts.max())
maxMem = unique[maxAttr]
restMem = np.delete(unique, maxAttr)
# print [member[-4:] for member in unique]

# while not pages[10].isEmpty():
# 	print ' '.join(pages[10].pop())
# print [pages[currentPage].size() for currentPage in range(0, totalPages)]

currentPage2 = 1
pages2 = {}
for i in range(0, totalPages):
	pages2[i] = i
for key, val in pages2.items():
	pages2[key] = Stack()
parser2 = UniqueParser()
parser2.feed(data)
for key, val in pages2.items():
	pages2[key] = reverse(pages2[key])

# while not pages2[10].isEmpty():
# 	print ' '.join(pages2[10].pop())
# print [pages2[currentPage2].size() for currentPage2 in range(0, totalPages)]

root = Element('processedDoc')
for i in range(0, totalPages):
	child = SubElement(root, 'page')
	child.set("no",str(i))
	m=0
	n=0
	while not pages2[i].isEmpty():
		lastPop = str(array(pages2[i].pop()))
		if lastPop in maxMem:
			text = []
			text2 = []
			body = SubElement(child, 'body')
			body.set("no",str(m))
			if pages2[i].peep() == 'span': dummy = 1
			else:
				text.append(' '.join(pages2[i].pop()))
				while (str(array(pages2[i].peep())) not in ['span','div','a']): 
					text.append(' '.join(pages2[i].pop()))
				body.text = ' '.join(text)
				m=m+1
		if lastPop in restMem:
			heading = SubElement(child,'heading')
			heading.set("feature",lastPop[-4:])
			heading.set("no",str(n))
			if pages2[i].peep() == 'span': dummy = 1
			else:
				heading.text = ' '.join(pages2[i].pop())
				n=n+1	

# with open(xml_file, "w") as text_file:
# 	text_file.write(tostring(root))
# os.remove(output_file)

from lxml import etree
doc = etree.XML(tostring(root))

def remove_empty_elements(doc):
  for element in doc.xpath('//*[not(node())]'):
    element.getparent().remove(element)

remove_empty_elements(doc)
x = etree.tostring(doc,pretty_print=True)

with open(xml_file, 'w') as the_file:
	the_file.write(x)