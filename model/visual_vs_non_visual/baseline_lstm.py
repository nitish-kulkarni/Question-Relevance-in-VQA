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
def train_test(X_train, Y_train, X_val, Y_val, X_test, Y_test, vocab_size, model_path, num_epochs):
	model = Sequential()
	model.add(Embedding(vocab_size, 10))
	model.add(LSTM(8, activation='sigmoid'))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

	for epoch in range(num_epochs):
		print "Training epoch : ",epoch
		i=0
		for item,label in zip(X_train, Y_train):
			i+=1
			if i%10000==0:
				print "Processed ",i
			model.fit(np.asarray([item]), np.asarray([[label]]), batch_size=1, verbose=0, epochs=1)
		model.save(model_path + 'epoch'+str(epoch)+'.h5')

		binpred=[]
		print "Predicting test"
		i=0
		for item in X_test:
			i+=1
			if i%10000==0:
				print "Processed test instances ",i
			pred = model.predict(np.asarray([item]), batch_size=1, verbose=0)
			binpred.append(pred[0][0]>0.07)
		
		binpred=np.asarray(binpred)
		bintest=(Y_test==1)
		#========== FOR CLASS Generic ====================
		print "For Class Generic (LSTM)"
		#Our Scores
		p=precision_score(bintest,binpred)
		print "precision: "+ str(p)
		r1=recall_score(bintest,binpred)
		print "recall: "+ str(r1)
		a1=accuracy_score(bintest,binpred)
		print "accuracy: "+str(a1)
		#========== FOR CLASS Specific ===================
		bintest_n=(Y_test==0)
		binpred_n=np.invert(binpred)
		print "For Class Specific (LSTM)"
		p=precision_score(bintest_n,binpred_n)
		print "precision: "+ str(p)
		r2=recall_score(bintest_n,binpred_n)
		print "recall: "+ str(r2)
		a2=accuracy_score(bintest,binpred)
		print "accuracy: "+str(a2)
		print "Normalized Accuracy : " + str((a1+a2)/2)

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
	val_file = params['val_file']
	test_file = params['test_file']
	model_location = params['model_location']
	num_epochs = params['num_epochs']

	unique_tags=['ADJ', 'ADP', 'ADV', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SPACE', 'SYM', 'VERB', 'X']
	vocab_size = len(unique_tags)

	X_train, Y_train = processData(train_file, unique_tags)
	X_val, Y_val = processData(val_file, unique_tags)
	X_test, Y_test = processData(test_file, unique_tags)

	train_test(X_train, Y_train, X_val, Y_val, X_test, Y_test, vocab_size, model_location, num_epochs)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', dest='train_file', type=str, default='./train.dat')
    parser.add_argument('--val_file', dest='val_file', type=str, default='./val.dat')
    parser.add_argument('--test_file', dest='test_file', type=str, default='./test.dat')
    parser.add_argument('--model_location', dest='model_location', type=str, default='./')
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=10)
    params = vars(parser.parse_args())
    main(params)