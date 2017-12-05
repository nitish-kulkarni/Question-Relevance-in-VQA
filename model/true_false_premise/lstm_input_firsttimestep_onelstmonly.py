import argparse
import numpy as np
import pandas as pd
from keras.preprocessing import sequence, text
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Merge, Input, RepeatVector, TimeDistributed, Concatenate, Reshape
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import precision_recall_fscore_support as score
import json
from sklearn.metrics import confusion_matrix
import torchfile

EMBEDDING_LEN = 300
MAX_LEN_SENTENCE = 10
BATCH_SIZE = 128

DATA_PATH = "./Data/"
QUESTIONS_MAP = json.load(open(DATA_PATH + 'vqa2_questions.json', 'r'))
QUESTIONS_MAP_QRPE = json.load(open(DATA_PATH + 'vqa1_questions.json', 'r'))
COCO_TRAIN_IMG_MAP = json.load(open("%s/coco_train_image_map.json" % DATA_PATH, "r"))
COCO_VAL_IMG_MAP = json.load(open("%s/coco_val_image_map.json" % DATA_PATH, "r"))
VG_IMG_MAP = json.load(open("%s/vg_image_map.json" % DATA_PATH, "r"))
### Collecting fc7 features for images
COCO_TRAIN_FEAT = torchfile.load("%s/train_fc7.t7" % DATA_PATH)
COCO_VAL_FEAT = torchfile.load("%s/val_fc7.t7" % DATA_PATH)
VG_FEAT = torchfile.load("%s/vg_fc7.t7" % DATA_PATH)
TRAIN_PATH = DATA_PATH + "train_firstorder_split_data.txt"
VAL_PATH = DATA_PATH + "val_firstorder_split_data.txt"
TEST_PATH = DATA_PATH + "val_firstorder_data.txt"
TRAIN_QRPE_PATH = DATA_PATH + "train_data.txt"
TEST_QRPE_PATH = DATA_PATH + "val_data.txt"
TRAIN_SECOND_PATH = DATA_PATH + "train_second_order_data.txt"
TEST_SECOND_PATH = DATA_PATH + "val_second_order_data.txt"
def get_image_feature(image_id, coco=True):
	if coco:
		return COCO_TRAIN_FEAT[COCO_TRAIN_IMG_MAP[image_id]] if image_id in COCO_TRAIN_IMG_MAP else COCO_VAL_FEAT[COCO_VAL_IMG_MAP[image_id]]
	else:
		return VG_FEAT[VG_IMG_MAP[image_id]]
def get_question(question_id, qrpe_flag):
	if qrpe_flag:
		return QUESTIONS_MAP_QRPE[question_id]
	return QUESTIONS_MAP[question_id]
def get_image_features(image_ids, coco):
	return np.array([get_image_feature(image_id, c) for image_id, c in zip(image_ids, coco)])
def get_questions(question_ids, qrpe_flag):
	return [get_question(qid, qrpe_flag) for qid in question_ids]

class Dataset():
	tokenizer = None
	data_train = None
	data_val = None
	data_test = None
	word_to_idx = None
	total_words = None
	qrpe_train = None
	qrpe_test = None
	second_train = None
	second_test = None

	def __init__(self, datapath):
		self.tokenizer = text.Tokenizer()
		self.data_train = pd.read_csv(TRAIN_PATH, sep="\t", header=None)
		self.data_train.columns = ["imgid","qid","rel","src"]
		self.data_val = pd.read_csv(VAL_PATH, sep="\t", header=None)
		self.data_val.columns = ["imgid","qid","rel","src"]
		self.data_test = pd.read_csv(TEST_PATH, sep="\t", header=None)
		self.data_test.columns = ["imgid","qid","rel","src"]
		self.qrpe_train = pd.read_csv(TRAIN_QRPE_PATH, sep="\t", header=None)
		self.qrpe_train.columns = ["imgid","qid","rel","src"]
		self.qrpe_test = pd.read_csv(TEST_QRPE_PATH, sep="\t", header=None)
		self.qrpe_test.columns = ["imgid","qid","rel","src"]
		self.second_train = pd.read_csv(TRAIN_SECOND_PATH, sep="\t", header=None)
		self.second_train.columns = ["imgid","qid","rel","src"]
		self.second_test = pd.read_csv(TEST_SECOND_PATH, sep="\t", header=None)
		self.second_test.columns = ["imgid","qid","rel","src"]
		# imgid for train, val, test
		self.data_train.imgid = self.data_train.imgid.astype(str)
		self.data_val.imgid = self.data_val.imgid.astype(str)
		self.data_test.imgid = self.data_test.imgid.astype(str)
		#imgid for qrpe train and test
		self.qrpe_train.imgid = self.qrpe_train.imgid.astype(str)
		self.qrpe_test.imgid = self.qrpe_test.imgid.astype(str)
		#imgid for second order train and test
		self.second_train.imgid = self.second_train.imgid.astype(str)
		self.second_test.imgid = self.second_test.imgid.astype(str)
		# qid for train, val, test
		self.data_train.qid = get_questions(self.data_train.qid.astype(str), False)
		self.data_val.qid = get_questions(self.data_val.qid.astype(str), False)
		self.data_test.qid = get_questions(self.data_test.qid.astype(str), False)
		#qid for qrpe train and test
		self.qrpe_train.qid = get_questions(self.qrpe_train.qid.astype(str), True)
		self.qrpe_test.qid = get_questions(self.qrpe_test.qid.astype(str), True)
		#qid for second order train and test
		self.second_train.qid = get_questions(self.second_train.qid.astype(str), False)
		self.second_test.qid = get_questions(self.second_test.qid.astype(str), False)
		# src for train, val, test
		self.data_train.src = self.data_train.src.astype(int)
		self.data_val.src = self.data_val.src.astype(int)
		self.data_test.src = self.data_test.src.astype(int)
		# src for qrpe
		self.qrpe_train.src = self.qrpe_train.src.astype(int)
		self.qrpe_test.src = self.qrpe_test.src.astype(int)
		# src for second_order
		self.second_train.src = self.second_train.src.astype(int)
		self.second_test.src = self.second_test.src.astype(int)
		self.tokenizer.fit_on_texts(list(self.data_train.qid.astype(str)) + list(self.data_val.qid.astype(str)) + list(self.data_test.qid.astype(str)) + list(self.qrpe_train.qid.astype(str)) + list(self.qrpe_test.qid.astype(str)) + list(self.second_train.qid.astype(str)) + list(self.second_test.qid.astype(str)))
		self.word_to_idx = self.tokenizer.word_index
		self.total_words = len(self.word_to_idx)

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

class DataGenerator():
	def __init__(self, batch_size = 128, shuffle = True):
		self.batch_size = batch_size
		self.shuffle = shuffle

	def __get_exploration_order(self, list_IDs):
		indexes = np.arange(len(list_IDs))
		if self.shuffle == True:
			np.random.shuffle(indexes)
		return indexes

	def __data_generation(self, list_IDs_temp, list_QIDs_temp, list_src_temp, tokenizer):
		X_img = get_image_features(list_IDs_temp, list_src_temp)
		X = tokenizer.texts_to_sequences(list_QIDs_temp)
		X_lang = sequence.pad_sequences(X, maxlen=MAX_LEN_SENTENCE)
		return X_lang, X_img

	def generate(self, labels, list_IDs, list_QIDs, list_src, tokenizer):
		while 1:
			indexes = self.__get_exploration_order(list_IDs)
			# Generate batches
			imax = int(len(indexes)/self.batch_size)
			for i in range(imax):
				indexes_temp = indexes[i*self.batch_size:(i+1)*self.batch_size]
				# Find list of IDs
				labels_temp = []
				list_IDs_temp = []
				list_QIDs_tmp = []
				list_src_temp = []
				for k in indexes_temp:
					list_IDs_temp.append(list_IDs[k])
					list_QIDs_tmp.append(list_QIDs[k])
					list_src_temp.append(int(list_src[k]))
					labels_temp.append(np.array(int(labels[k])))
				# Generate data
				X_lang, X_img = self.__data_generation(list_IDs_temp, list_QIDs_tmp, list_src_temp, tokenizer)
				yield [X_lang, X_img], np.array(labels_temp)

class LSTMModel():
	def build_model(self, num_vocab, embedding_matrix, max_len):
		lang_input = Input(shape=(max_len,))
		loutx = Embedding(input_dim=num_vocab, output_dim=EMBEDDING_LEN, weights=[embedding_matrix], input_length=max_len)(lang_input)

		img_input = Input(shape=(4096,))
		i1x = Dense(EMBEDDING_LEN, input_dim = 4096, activation='relu')(img_input)
		ioutx = Reshape((1,300), input_shape=(300,))(i1x)

		x = Concatenate(axis=1)([ioutx, loutx])
		x = LSTM(512,return_sequences=False)(x)
		x = Dense(100, activation='relu', W_regularizer=l2(0.0001), b_regularizer=l2(0.0001))(x)
		x = Dense(50, activation='relu', W_regularizer=l2(0.0001), b_regularizer=l2(0.0001))(x)
		xout = Dense(1, activation='sigmoid', W_regularizer=l2(0.0001), b_regularizer=l2(0.0001))(x)
		
		model = Model(inputs=[lang_input, img_input], outputs=xout)
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		print(model.summary())
		return model

### Main function which trains the model, tests the model and report metrics
def main(params):
	datapath = params["datapath"]
	embeddings_path = params["embeddings_path"]

	Ds = Dataset(datapath)
	embedding_matrix = Ds.create_embedding_matrix(embeddings_path)
	print "Obtained embeddings"
	num_vocab = Ds.total_words + 1

	lm = LSTMModel()
	model = lm.build_model(num_vocab, embedding_matrix, MAX_LEN_SENTENCE)
	print "Built Model"
	print "Training now..."
	training_generator = DataGenerator(batch_size=100).generate(Ds.data_train.rel, Ds.data_train.imgid, Ds.data_train.qid, Ds.data_train.src, Ds.tokenizer)
	validation_generator = DataGenerator(batch_size=125).generate(Ds.data_val.rel, Ds.data_val.imgid, Ds.data_val.qid, Ds.data_val.src, Ds.tokenizer)
	#model.fit_generator(epochs=1, generator=training_generator, validation_data=validation_generator, steps_per_epoch=(len(Ds.data_train.imgid)/100), validation_steps = len(Ds.data_val.imgid)/125, verbose=1)
	model.fit_generator(epochs=10, generator=training_generator, validation_data=validation_generator, steps_per_epoch=(len(Ds.data_train.imgid)/100), validation_steps = len(Ds.data_val.imgid)/125, verbose=1)

	### Defining generators for prediction
	training_generator = DataGenerator(batch_size=100, shuffle=False).generate(Ds.data_train.rel, Ds.data_train.imgid, Ds.data_train.qid, Ds.data_train.src, Ds.tokenizer)
	validation_generator = DataGenerator(batch_size=125, shuffle=False).generate(Ds.data_val.rel, Ds.data_val.imgid, Ds.data_val.qid, Ds.data_val.src, Ds.tokenizer)
	testing_generator = DataGenerator(batch_size=8, shuffle=False).generate(Ds.data_test.rel, Ds.data_test.imgid, Ds.data_test.qid, Ds.data_test.src, Ds.tokenizer)
	qrpe_training_generator = DataGenerator(batch_size=116, shuffle=False).generate(Ds.qrpe_train.rel, Ds.qrpe_train.imgid, Ds.qrpe_train.qid, Ds.qrpe_train.src, Ds.tokenizer)
	qrpe_testing_generator = DataGenerator(batch_size=115, shuffle=False).generate(Ds.qrpe_test.rel, Ds.qrpe_test.imgid, Ds.qrpe_test.qid, Ds.qrpe_test.src, Ds.tokenizer)
	second_training_generator = DataGenerator(batch_size=80, shuffle=False).generate(Ds.second_train.rel, Ds.second_train.imgid, Ds.second_train.qid, Ds.second_train.src, Ds.tokenizer)
	second_testing_generator = DataGenerator(batch_size=21, shuffle=False).generate(Ds.second_test.rel, Ds.second_test.imgid, Ds.second_test.qid, Ds.second_test.src, Ds.tokenizer)

	### Testing on first order train dataset
	pred = model.predict_generator(generator=training_generator, steps=(len(Ds.data_train.imgid)/100), verbose=1)
	fid = open(datapath + "firstorder_train_pred_inputfirsttimestep_onelstm.txt", "w")
	for p in pred:
		fid.write(str(p[0])+"\n")
	fid.close()
	precision, recall, fscore, support = score(Ds.data_train.rel, pred.round(), labels=[0, 1])
	print "Metrics on first order train dataset"
	print('precision: {}'.format(precision))
	print('recall: {}'.format(recall))
	print('fscore: {}'.format(fscore))
	print('support: {}'.format(support))
	print confusion_matrix(Ds.data_train.rel, pred.round())

	### Testing on first order val dataset
	pred = model.predict_generator(generator=validation_generator, steps=(len(Ds.data_val.imgid)/125), verbose=1)
	fid = open(datapath + "firstorder_val_pred_inputfirsttimestep_onelstm.txt", "w")
	for p in pred:
		fid.write(str(p[0])+"\n")
	fid.close()
	precision, recall, fscore, support = score(Ds.data_val.rel, pred.round(), labels=[0, 1])
	print "Metrics on first order val dataset"
	print('precision: {}'.format(precision))
	print('recall: {}'.format(recall))
	print('fscore: {}'.format(fscore))
	print('support: {}'.format(support))
	print confusion_matrix(Ds.data_val.rel, pred.round())

	### Testing on first order test dataset
	pred = model.predict_generator(generator=testing_generator, steps=(len(Ds.data_test.imgid)/8), verbose=1)
	fid = open(datapath + "firstorder_test_pred_inputfirsttimestep_onelstm.txt", "w")
	for p in pred:
		fid.write(str(p[0])+"\n")
	fid.close()
	precision, recall, fscore, support = score(Ds.data_test.rel, pred.round(), labels=[0, 1])
	print "Metrics on first order test dataset"
	print('precision: {}'.format(precision))
	print('recall: {}'.format(recall))
	print('fscore: {}'.format(fscore))
	print('support: {}'.format(support))
	print confusion_matrix(Ds.data_test.rel, pred.round())

	### Testing on qrpe train dataset
	pred = model.predict_generator(generator=qrpe_training_generator, steps=(len(Ds.qrpe_train.imgid)/116), verbose=1)
	fid = open(datapath + "qrpe_train_pred_inputfirsttimestep_onelstm.txt", "w")
	for p in pred:
		fid.write(str(p[0])+"\n")
	fid.close()
	precision, recall, fscore, support = score(Ds.qrpe_train.rel, pred.round(), labels=[0, 1])
	print "Metrics on qrpe train dataset"
	print('precision: {}'.format(precision))
	print('recall: {}'.format(recall))
	print('fscore: {}'.format(fscore))
	print('support: {}'.format(support))
	print confusion_matrix(Ds.qrpe_train.rel, pred.round())

	### Testing on qrpe test dataset
	pred = model.predict_generator(generator=qrpe_testing_generator, steps=(len(Ds.qrpe_test.imgid)/115), verbose=1)
	fid = open(datapath + "qrpe_test_pred_inputfirsttimestep_onelstm.txt", "w")
	for p in pred:
		fid.write(str(p[0])+"\n")
	fid.close()
	precision, recall, fscore, support = score(Ds.qrpe_test.rel, pred.round(), labels=[0, 1])
	print "Metrics on qrpe test dataset"
	print('precision: {}'.format(precision))
	print('recall: {}'.format(recall))
	print('fscore: {}'.format(fscore))
	print('support: {}'.format(support))
	print confusion_matrix(Ds.qrpe_test.rel, pred.round())

	### Testing on second order train dataset
	pred = model.predict_generator(generator=second_training_generator, steps=(len(Ds.second_train.imgid)/80), verbose=1)
	fid = open(datapath + "second_train_pred_inputfirsttimestep_onelstm.txt", "w")
	for p in pred:
		fid.write(str(p[0])+"\n")
	fid.close()
	precision, recall, fscore, support = score(Ds.second_train.rel, pred.round(), labels=[0, 1])
	print "Metrics on second train dataset"
	print('precision: {}'.format(precision))
	print('recall: {}'.format(recall))
	print('fscore: {}'.format(fscore))
	print('support: {}'.format(support))
	print confusion_matrix(Ds.second_train.rel, pred.round())

	### Testing on second order test dataset
	pred = model.predict_generator(generator=second_testing_generator, steps=(len(Ds.second_test.imgid)/21), verbose=1)
	fid = open(datapath + "second_test_pred_inputfirsttimestep_onelstm.txt", "w")
	for p in pred:
		fid.write(str(p[0])+"\n")
	fid.close()
	precision, recall, fscore, support = score(Ds.second_test.rel, pred.round(), labels=[0, 1])
	print "Metrics on second test dataset"
	print('precision: {}'.format(precision))
	print('recall: {}'.format(recall))
	print('fscore: {}'.format(fscore))
	print('support: {}'.format(support))
	print confusion_matrix(Ds.second_test.rel, pred.round())

if __name__=='__main__':
	### Read user inputs
	parser = argparse.ArgumentParser()
	parser.add_argument("--datapath", dest="datapath", type=str, default="./")
	parser.add_argument("--embeddings_path", dest="embeddings_path", type=str, default="./glove.840B.300d.txt")
	params = vars(parser.parse_args())
	main(params)