from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import argparse
import numpy as np

### Define and train an lstm model
def train_test(X_train, Y_train, X_test, Y_test, vocab_size, model_path, pre_trained_model):
	model = Sequential()
	model.add(Embedding(vocab_size, 10))
	model.add(LSTM(8, activation='sigmoid'))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

	model.load_weights(pre_trained_model)
	binpred=[]
	print "Predicting test"
	i=0
	for item in X_test:
		i+=1
		if i%20000==0:
			print "Processed test instances ",i
			break
		pred = model.predict(np.asarray([item]), batch_size=1, verbose=0)
		binpred.append(pred[0][0])
	
	binpred=np.asarray(binpred)
	with open('predictions.dat','w') as f:
		for i in range(len(binpred)):
			f.write(str(Y_test[i])+','+str(binpred[i])+'\n')

### Function which takes in a file path as input and 
### fetches only pos tags as features for every instance
### along with the label and outputs a numpy matrix for
### features and labels
def processData(file_path, unique_tags):
	question_tags = []
	labels = []
	with open(file_path, 'r') as f:
		for line in f:
			line = line.strip('\n')
			(label, question) = line.split("\t")
			if label=="N":
				labels.append(0)
			else:
				labels.append(1)
			question_split = question.split("|")
			odd_number_check = 0
			taglist = []
			for token in question_split:
				if odd_number_check == 0:
					odd_number_check = 1
				else:
					if token in unique_tags:
						index = unique_tags.index(token)
						taglist.append(index)
					odd_number_check = 0
			question_tags.append(taglist)
	X = np.asarray(question_tags)
	Y = np.asarray(labels)
	return X,Y

def main(params):
	train_file = params['train_file']
	test_file = params['test_file']
	model_location = params['model_location']
	pre_trained_model = params['pre_trained_model']

	unique_tags=['ADJ', 'ADP', 'ADV', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SPACE', 'SYM', 'VERB', 'X']
	vocab_size = len(unique_tags)

	X_train, Y_train = processData(train_file, unique_tags)
	X_test, Y_test = processData(test_file, unique_tags)

	train_test(X_train, Y_train, X_test, Y_test, vocab_size, model_location, pre_trained_model)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', dest='train_file', type=str, default='./train.dat')
    parser.add_argument('--test_file', dest='test_file', type=str, default='./test.dat')
    parser.add_argument('--model_location', dest='model_location', type=str, default='./')
    parser.add_argument('--pre_trained_model', dest='pre_trained_model', type=str, default='./')
    params = vars(parser.parse_args())
    main(params)