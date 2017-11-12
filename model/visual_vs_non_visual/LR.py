import sys
from math import exp,log
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import numpy as np

unique_tags=['ADJ', 'ADP', 'ADV', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SPACE', 'SYM', 'VERB', 'X']
n = len(unique_tags)
vocab_size = n
eta = float(sys.argv[1])
mu = float(sys.argv[2])
traindata_size = int(sys.argv[3])
threshold = float(sys.argv[4])
train_data = sys.argv[5]
val_data = sys.argv[6]
test_data = sys.argv[7]

def sigmoid(score):
	if score > 20.0:
		score = 20.0
	elif score < -20.0:
		score = -20.0
	expv = exp(score)
	return expv / (1 + expv)

def get_lidx(label):
	if label=="N":
		return 0.0
	else:
		return 1.0

def get_hash_idx(curDoc):
	question_split = curDoc.split("|")
	odd_number_check = 0
	x = {}
	for token in question_split:
		if odd_number_check == 0:
			odd_number_check = 1
		else:
			if token in unique_tags:
				index = unique_tags.index(token)
				if index in x:
					x[index] += 1.0
				else:
					x[index] = 1.0
			odd_number_check = 0
	return x

def train():
	k=0
	A = [0.0 for i in range(vocab_size)]
	B = [0.0 for j in range(vocab_size)]
	t = 0.0
	lambdav = 0.0
	c = 0.0
	lcl = 0.0

	for line in sys.stdin:
		## Fetch the document
		(label, curDoc) = line.strip("\n").split("\t")

		## Update k and lambda
		k += 1
		if (k%traindata_size==1):
			t+=1.0
			lambdav = eta
			c = (1.0-(2.0 * lambdav * mu))
			lcl = 0.0

		## Obtain y
		y = get_lidx(label)

		## Tokenize
		tokens_hash = get_hash_idx(curDoc)

		## Obtain p
		p = 0.0
		for tokenidx in tokens_hash:
			p += (B[tokenidx]*tokens_hash[tokenidx])
		prob = sigmoid(p)
		p = lambdav * (y - sigmoid(p))
		lcl += (y*log(prob) + (1.0-y)*log(1-prob))

		for tokenidx in tokens_hash:
			val = (c ** (k - A[tokenidx]))
			B[tokenidx] = B[tokenidx] * val
			B[tokenidx] += (p * tokens_hash[tokenidx])
			A[tokenidx] = k

	for j in range(vocab_size):
		if A[j]!=0.0:
			val = c ** (k - A[j])
			B[j] = B[j] * val
	return B

def test(B, data_file):
	tp = 0.0
	tn = 0.0
	fp = 0.0
	fn = 0.0
	bintest1 = []
	binpred1 = []
	bintest0 = []
	binpred0 = []
	with open(data_file, "r") as f:
		for line in f:
			(label, curDoc) = line.strip("\n").split("\t")
			tokens_hash = get_hash_idx(curDoc)
			p = 0.0
			for tokenidx in tokens_hash:
				p += (B[tokenidx] * tokens_hash[tokenidx])
			p = sigmoid(p)
			y = get_lidx(label)
			bintest1.append(y)
			bintest0.append(1-y)
			if p<threshold:
				binpred1.append(0)
				binpred0.append(1)
			else:
				binpred1.append(1)
				binpred0.append(0)
			if label=="N" and p<threshold:
				tn += 1.0
			elif label=="N" and p>=threshold:
				fp += 1.0
			elif label=="V" and p>=threshold:
				tp += 1.0
			else:
				fn += 1.0
	binpred1 = np.asarray(binpred1)
	bintest1 = np.asarray(bintest1)
	binpred0 = np.asarray(binpred0)
	bintest0 = np.asarray(bintest0)
	print "For Class Generic (LSTM)"
	pr1 = precision_score(bintest1,binpred1)
	print "precision: " + str(pr1)
	re1 = recall_score(bintest1,binpred1)
	print "recall: " + str(re1)
	print "For Class Specific (LSTM)"
	pr2 = precision_score(bintest0,binpred0)
	print "precision: " + str(pr2)
	re2 = recall_score(bintest0,binpred0)
	print "recall: " + str(re2)
	print "Normalized Accuracy : " + str((re1 + re2)/2)
	sys.stdout.write("TP: "+str(tp) + ", TN: " + str(tn) + ", FP: " + str(fp) + ", FN: " + str(fn) + "\n")

B = train()
test(B, train_data)
test(B, val_data)
test(B, test_data)