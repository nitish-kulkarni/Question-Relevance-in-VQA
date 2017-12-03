import argparse
import numpy as np
import pandas as pd
from keras.preprocessing import sequence, text
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Merge, Input, RepeatVector, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import precision_recall_fscore_support as score
import json
from sklearn.metrics import confusion_matrix

EMBEDDING_LEN = 300

class Dataset():
	tokenizer = None
	data_train = None
	data_val = None
	data_test = None
	word_to_idx = None
	total_words = None
	qrpe_train = None
	qrpe_test = None

	def __init__(self, datapath):
		self.tokenizer = text.Tokenizer()
		self.data_train = pd.read_csv(datapath + "processed_train_data.txt", sep="\t", header=None)
		self.data_train.columns = ["img_fts","ques","rel"]
		self.data_val = pd.read_csv(datapath + "processed_val_data.txt", sep="\t", header=None)
		self.data_val.columns = ["img_fts","ques","rel"]
		self.data_test = pd.read_csv(datapath + "processed_test_data.txt", sep="\t", header=None)
		self.data_test.columns = ["img_fts","ques","rel"]
		self.qrpe_train = pd.read_csv(datapath + "processed_qrpe_train_data.txt", sep="\t", header=None)
		self.qrpe_train.columns = ["img_fts","ques","rel"]
		self.qrpe_test = pd.read_csv(datapath + "processed_qrpe_test_data.txt", sep="\t", header=None)
		self.qrpe_test.columns = ["img_fts","ques","rel"]
		self.data_train.ques = self.data_train.ques.astype(str)
		self.data_val.ques = self.data_val.ques.astype(str)
		self.data_test.ques = self.data_test.ques.astype(str)
		self.tokenizer.fit_on_texts(list(self.data_train.ques) + list(self.data_val.ques) + list(self.data_test.ques))
		self.word_to_idx = self.tokenizer.word_index
		self.total_words = len(self.word_to_idx)

	def process_dataframe(self, inpdata, max_len_sentence):
		X = self.tokenizer.texts_to_sequences(inpdata.ques)
		X_lang = sequence.pad_sequences(X, maxlen=max_len_sentence)
		X_img = np.array([np.fromstring(key, dtype=np.float32, sep=",").reshape(300) for key in inpdata.img_fts])
		Y = inpdata.rel.astype(int)
		return X_lang, X_img, Y

	def create_embedding_matrix(self, embeddings_path):
		embeddings = {}
		with open(embeddings_path) as f:
			for line in f:
				values = line.split()
				embedding = np.asarray(values[1:], dtype='float32')
				embeddings[values[0]] = embedding
		sz_embedding_mat = self.total_words + 1
		embedding_matrix = np.zeros((sz_embedding_mat, EMBEDDING_LEN))
		for key in self.word_to_idx:
			if key in embeddings:
				embedding_matrix[self.word_to_idx[key]] = embeddings[key]
		print "Initialized word embeddings"
		return embedding_matrix

class LSTMModel():
	def build_model(self, num_vocab, embedding_matrix, max_len):
		lang_model = Sequential()
		lang_model.add(Embedding(input_dim=num_vocab, output_dim=EMBEDDING_LEN, weights=[embedding_matrix], input_length=max_len))
		lang_model.add(LSTM(256,return_sequences=True))
		lang_model.add(TimeDistributed(Dense(EMBEDDING_LEN)))

		image_model = Sequential()
		image_model.add(Dense(EMBEDDING_LEN, input_dim = 300, activation='relu'))
		image_model.add(RepeatVector(max_len))

		model = Sequential()
		model.add(Merge([lang_model, image_model], mode='concat'))
		model.add(LSTM(1000,return_sequences=False, input_shape=()))
		model.add(Dense(100, activation='relu', W_regularizer=l2(0.0001), b_regularizer=l2(0.0001)))
		model.add(Dense(50, activation='relu', W_regularizer=l2(0.0001), b_regularizer=l2(0.0001)))
		model.add(Dense(1, activation='sigmoid', W_regularizer=l2(0.0001), b_regularizer=l2(0.0001)))

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
	X_train_lang, X_train_img, Y_train = Ds.process_dataframe(Ds.data_train, max_len_sentence)
	X_val_lang, X_val_img, Y_val = Ds.process_dataframe(Ds.data_val, max_len_sentence)
	print "Obtained processed training and validation data"
	embedding_matrix = Ds.create_embedding_matrix(embeddings_path)
	print "Obtained embeddings"
	num_vocab = Ds.total_words + 1

	lm = LSTMModel()
	model = lm.build_model(num_vocab, embedding_matrix, max_len_sentence)
	print "Built Model"
	print "Training now..."
	#filepath = model_path + "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
	#checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	#callbacks_list = [checkpoint]
	model.fit(x=[X_train_lang, X_train_img], y=Y_train, batch_size=128, epochs=5, verbose=1, shuffle=True, callbacks=None, validation_data=([X_val_lang, X_val_img], Y_val))

	### Testing on first order train dataset
	X_lang, X_img, Y = Ds.process_dataframe(Ds.data_train, max_len_sentence)
	pred = model.predict([X_lang, X_img], batch_size=32, verbose=0)
	fid = open(datapath + "firstorder_train_pred.txt", "w")
	for p in pred:
		fid.write(str(p[0])+"\n")
	fid.close()
	precision, recall, fscore, support = score(Y, pred.round(), labels=[0, 1])
	print "Metrics on qrpe test dataset"
	print('precision: {}'.format(precision))
	print('recall: {}'.format(recall))
	print('fscore: {}'.format(fscore))
	print('support: {}'.format(support))
	print confusion_matrix(Y, pred.round())

	### Testing on first order val dataset
	X_lang, X_img, Y = Ds.process_dataframe(Ds.data_val, max_len_sentence)
	pred = model.predict([X_lang, X_img], batch_size=32, verbose=0)
	fid = open(datapath + "firstorder_val_pred.txt", "w")
	for p in pred:
		fid.write(str(p[0])+"\n")
	fid.close()
	precision, recall, fscore, support = score(Y, pred.round(), labels=[0, 1])
	print "Metrics on qrpe test dataset"
	print('precision: {}'.format(precision))
	print('recall: {}'.format(recall))
	print('fscore: {}'.format(fscore))
	print('support: {}'.format(support))
	print confusion_matrix(Y, pred.round())

	### Testing on first order test dataset
	X_lang, X_img, Y = Ds.process_dataframe(Ds.data_test, max_len_sentence)
	pred = model.predict([X_lang, X_img], batch_size=32, verbose=0)
	fid = open(datapath + "firstorder_test_pred.txt", "w")
	for p in pred:
		fid.write(str(p[0])+"\n")
	fid.close()
	precision, recall, fscore, support = score(Y, pred.round(), labels=[0, 1])
	print "Metrics on qrpe test dataset"
	print('precision: {}'.format(precision))
	print('recall: {}'.format(recall))
	print('fscore: {}'.format(fscore))
	print('support: {}'.format(support))
	print confusion_matrix(Y, pred.round())

	### Testing on qrpe train dataset
	X_lang, X_img, Y = Ds.process_dataframe(Ds.qrpe_train, max_len_sentence)
	pred = model.predict([X_lang, X_img], batch_size=32, verbose=0)
	fid = open(datapath + "qrpe_train_pred.txt", "w")
	for p in pred:
		fid.write(str(p[0])+"\n")
	fid.close()
	precision, recall, fscore, support = score(Y, pred.round(), labels=[0, 1])
	print "Metrics on qrpe test dataset"
	print('precision: {}'.format(precision))
	print('recall: {}'.format(recall))
	print('fscore: {}'.format(fscore))
	print('support: {}'.format(support))
	print confusion_matrix(Y, pred.round())

	### Testing on qrpe test dataset
	X_lang, X_img, Y = Ds.process_dataframe(Ds.qrpe_test, max_len_sentence)
	pred = model.predict([X_lang, X_img], batch_size=32, verbose=0)
	fid = open(datapath + "qrpe_test_pred.txt", "w")
	for p in pred:
		fid.write(str(p[0])+"\n")
	fid.close()
	precision, recall, fscore, support = score(Y, pred.round(), labels=[0, 1])
	print "Metrics on qrpe test dataset"
	print('precision: {}'.format(precision))
	print('recall: {}'.format(recall))
	print('fscore: {}'.format(fscore))
	print('support: {}'.format(support))
	print confusion_matrix(Y, pred.round())

if __name__=='__main__':
	### Read user inputs
	parser = argparse.ArgumentParser()
	parser.add_argument("--datapath", dest="datapath", type=str, default="./")
	parser.add_argument("--train_data_split", dest="train_data_split", type=float, default=0.8)
	parser.add_argument("--max_len_sentence", dest="max_len_sentence", type=int, default=10)
	parser.add_argument("--embeddings_path", dest="embeddings_path", type=str, default="./glove.840B.300d.txt")
	parser.add_argument("--model_path", dest="model_path", type=str, default="./models/")
	params = vars(parser.parse_args())
	main(params)