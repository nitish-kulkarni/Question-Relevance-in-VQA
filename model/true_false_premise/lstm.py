import argparse
import numpy as np
import pandas as pd
from keras.preprocessing import sequence, text
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, merge, Input 
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import precision_recall_fscore_support as score
import json

EMBEDDING_LEN = 300

class Dataset():
	tokenizer = None
	data_train = None
	data_val = None
	data_test = None
	word_to_idx = None
	total_words = None
	num_imgs_train = None
	num_imgs_val = None
	num_imgs_test = None

	def __init__(self, datapath):
		self.tokenizer = text.Tokenizer()
		self.data_train = pd.read_csv(datapath + "processed_train_data.txt", sep="\t", header=None)
		self.data_train.columns = ["img_fts","ques","rel"]
		self.data_val = pd.read_csv(datapath + "processed_val_data.txt", sep="\t", header=None)
		self.data_val.columns = ["img_fts","ques","rel"]
		self.data_test = pd.read_csv(datapath + "processed_test_data.txt", sep="\t", header=None)
		self.data_test.columns = ["img_fts","ques","rel"]
		self.data_train.ques = self.data_train.ques.astype(str)
		self.data_val.ques = self.data_val.ques.astype(str)
		self.data_test.ques = self.data_test.ques.astype(str)
		self.tokenizer.fit_on_texts(list(self.data_train.ques) + list(self.data_val.ques) + list(self.data_test.ques))
		self.word_to_idx = self.tokenizer.word_index
		self.total_words = len(self.word_to_idx)
		self.num_imgs_train = len(self.data_train.img_fts)
		self.num_imgs_val = len(self.data_val.img_fts)
		self.num_imgs_test = len(self.data_test.img_fts)

	def process_dataframe(self, inpdata, max_len_sentence, flag):
		X = self.tokenizer.texts_to_sequences(inpdata.ques)
		X = sequence.pad_sequences(X, maxlen=max_len_sentence)
		if flag==0:
			X = np.concatenate((np.expand_dims(range(self.total_words+1,self.total_words+1+self.num_imgs_train),1),X),axis=1)
		elif flag==1:
			X = np.concatenate((np.expand_dims(range(self.total_words+1+self.num_imgs_train,self.total_words+1+self.num_imgs_train+self.num_imgs_val),1),X),axis=1)
		else:
			X = np.concatenate((np.expand_dims(range(self.total_words+1+self.num_imgs_train+self.num_imgs_val,self.total_words+1+self.num_imgs_train+self.num_imgs_val+self.num_imgs_val),1),X),axis=1)
		Y = inpdata.rel.astype(int)
		return X,Y

	def create_embedding_matrix(self, embeddings_path):
		embeddings = {}
		with open(embeddings_path) as f:
			for line in f:
				values = line.split()
				embedding = np.asarray(values[1:], dtype='float32')
				embeddings[values[0]] = embedding
		sz_embedding_mat = self.total_words + 1 + self.num_imgs_train + self.num_imgs_val + self.num_imgs_test
		embedding_matrix = np.zeros((sz_embedding_mat, EMBEDDING_LEN))
		for key in self.word_to_idx:
			if key in embeddings:
				embedding_matrix[self.word_to_idx[key]] = embeddings[key]
		print "Initialized word embeddings"
		embedding_matrix[self.total_words+1:self.total_words+1+self.num_imgs_train] = np.array([np.fromstring(key, dtype=np.float16, sep=",").reshape(300) for key in self.data_train.img_fts])
		print "Initialized train features as embeddings"
		embedding_matrix[self.total_words+1+self.num_imgs_train:self.total_words+1+self.num_imgs_train+self.num_imgs_val] = np.array([np.fromstring(key, dtype=np.float16, sep=",").reshape(300) for key in self.data_val.img_fts])
		print "Initialized val features as embeddings"
		embedding_matrix[self.total_words+1+self.num_imgs_train+self.num_imgs_val:self.total_words+1+self.num_imgs_train+self.num_imgs_val+self.num_imgs_test] = np.array([np.fromstring(key, dtype=np.float16, sep=",").reshape(300) for key in self.data_test.img_fts])
		print "Initialized test features as embeddings"
		return embedding_matrix

class LSTMModel():
	def build_model(self, num_vocab, embedding_matrix, max_len):
		lstm = Sequential()
		lstm.add(Embedding(input_dim=num_vocab, output_dim=EMBEDDING_LEN, weights=[embedding_matrix], input_length=max_len, trainable=False))
		lstm.add(LSTM(256, dropout_W=0.5, dropout_U=0.5))
		lstm.add(Dense(100, activation='relu', W_regularizer=l2(0.0001), b_regularizer=l2(0.0001)))
		lstm.add(Dense(50, activation='relu', W_regularizer=l2(0.0001), b_regularizer=l2(0.0001)))
		lstm.add(Dense(1, activation='sigmoid', W_regularizer=l2(0.0001), b_regularizer=l2(0.0001)))
		l_input = Input(shape=(max_len,))
		prediction = lstm(l_input)
		model = Model(input=[l_input], output=prediction)
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		print(model.summary())
		return model

### Main function which trains the model, tests the model and report metrics
def main(params):
	datapath = params["datapath"]
	train_data_split = params["train_data_split"]
	max_len_sentence = params["max_len_sentence"]
	embeddings_path = params["embeddings_path"]
	model_path = params["model_path"]

	Ds = Dataset(datapath)
	X_train, Y_train = Ds.process_dataframe(Ds.data_train, max_len_sentence, 0)
	X_val, Y_val = Ds.process_dataframe(Ds.data_val, max_len_sentence, 1)
	print "Obtained processed training and validation data"
	embedding_matrix = Ds.create_embedding_matrix(embeddings_path)
	print "Obtained embeddings"
	num_vocab = Ds.total_words + 1 + Ds.num_imgs_train + Ds.num_imgs_val + Ds.num_imgs_test

	lm = LSTMModel()
	model = lm.build_model(num_vocab, embedding_matrix, max_len_sentence+1)
	print "Built Model"
	print "Training now..."
	filepath = model_path + "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]
	model.fit(x=[X_train], y=Y_train, batch_size=128, epochs=30, verbose=1, shuffle=True, callbacks=callbacks_list, validation_data=(X_val, Y_val))

	X_test, Y_test = Ds.process_dataframe(test_data, max_len_sentence, 2)
	pred = model.predict(X_test, batch_size=32, verbose=0)
	precision, recall, fscore, support = score(Y_test, pred.round(), labels=[0, 1])

	print "Metrics on test dataset"
	print('precision: {}'.format(precision))
	print('recall: {}'.format(recall))
	print('fscore: {}'.format(fscore))
	print('support: {}'.format(support))

if __name__=='__main__':
	### Read user inputs
	parser = argparse.ArgumentParser()
	parser.add_argument("--datapath", dest="datapath", type=str, default="../../../project_nongit/Data/")
	parser.add_argument("--train_data_split", dest="train_data_split", type=float, default=0.8)
	parser.add_argument("--max_len_sentence", dest="max_len_sentence", type=int, default=10)
	parser.add_argument("--embeddings_path", dest="embeddings_path", type=str, default="../../../../11-785/Project/Data/glove.840B.300d.txt")
	parser.add_argument("--model_path", dest="model_path", type=str, default="./models/")
	params = vars(parser.parse_args())
	main(params)