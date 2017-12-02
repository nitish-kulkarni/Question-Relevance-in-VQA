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
import ast

EMBEDDING_LEN = 300

class Dataset():
	tokenizer = None
	data = None
	word_to_idx = None
	total_words = None
	num_imgs = None

	def __init__(self, datapath):
		self.tokenizer = text.Tokenizer()
		self.data = pd.read_json(datapath)
		self.data.question = self.data.question.astype(str)
		self.tokenizer.fit_on_texts(list(self.data.question))
		self.word_to_idx = self.tokenizer.word_index
		self.total_words = len(self.word_to_idx)
		self.num_imgs = len(self.data.image_id)
		idx = self.total_words + 1
		new_idx = np.expand_dims((np.array(range(1,self.num_imgs+1))+idx).T,axis=1)
		self.data.image_id = new_idx

	def create_dataset(self, train_data_split):
		total_data_instances = len(self.data)
		# Shuffle the data indexes
		np.random.seed(42)
		perm = np.random.permutation(self.data.index)
		train_end_idx = int(train_data_split*total_data_instances)
		# Create train and test based on input split
		train_data = self.data.iloc[perm[0:train_end_idx]]
		test_data = self.data.iloc[perm[train_end_idx:total_data_instances]]
		print "Number of train data instances read", len(train_data)
		print "Number of test data instances read", len(test_data)
		return train_data, test_data

	def process_dataframe(self, inpdata, max_len_sentence):
		X = self.tokenizer.texts_to_sequences(inpdata.question)
		X = sequence.pad_sequences(X, maxlen=max_len_sentence)
		X = np.concatenate((np.expand_dims(inpdata.image_id,1),X),axis=1)
		Y = inpdata.relevance.astype(int)
		return X,Y

	def create_embedding_matrix(self, embeddings_path):
		embeddings = {}
		with open(embeddings_path) as f:
			for line in f:
				values = line.split()
				embedding = np.asarray(values[1:], dtype='float32')
				embeddings[values[0]] = embedding
		sz_embedding_mat = self.total_words + 1 + self.num_imgs
		embedding_matrix = np.zeros((sz_embedding_mat, EMBEDDING_LEN))
		for key in self.word_to_idx:
			if key in embeddings:
				embedding_matrix[self.word_to_idx[key]] = embeddings[key]
		for i in range(self.total_words+1, sz_embedding_mat):
			embedding_matrix[i] = np.array(ast.literal_eval(self.data.image_features_fc7[i-self.total_words-1]))
		return embedding_matrix

class LSTMModel():
	def build_model(self, num_vocab, embedding_matrix, max_len):
		lstm = Sequential()
		lstm.add(Embedding(input_dim=num_vocab, output_dim=EMBEDDING_LEN, \
			weights=[embedding_matrix], input_length=max_len, trainable=False))
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
	train_data, test_data = Ds.create_dataset(train_data_split)
	X_train, Y_train = Ds.process_dataframe(train_data, max_len_sentence)
	# Storage reduction
	train_data = None
	print "Obtained processed training data"
	embedding_matrix = Ds.create_embedding_matrix(embeddings_path)
	print "Obtained embeddings"
	num_vocab = Ds.total_words + 1 + Ds.num_imgs

	lm = LSTMModel()
	model = lm.build_model(num_vocab, embedding_matrix, max_len_sentence+1)
	print "Built Model"
	print "Training now..."
	filepath = model_path + "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]
	model.fit(x=[X_train], y=Y_train, batch_size=128, epochs=30, verbose=1, validation_split=0.2, shuffle=True, callbacks=callbacks_list)

	X_test, Y_test = Ds.process_dataframe(test_data, max_len_sentence)
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
	#parser.add_argument("--datapath", dest="datapath", type=str, default="../../Data/quora_duplicate_questions.tsv")
	parser.add_argument("--datapath", dest="datapath", type=str, default="data_transformed.json")
	parser.add_argument("--train_data_split", dest="train_data_split", type=float, default=0.8)
	parser.add_argument("--max_len_sentence", dest="max_len_sentence", type=int, default=10)
	parser.add_argument("--embeddings_path", dest="embeddings_path", type=str, default="glove.840B.300d.txt")
	parser.add_argument("--model_path", dest="model_path", type=str, default="./models/")
	params = vars(parser.parse_args())
	main(params)