import os
import time
import numpy as np
from ast import literal_eval
from sklearn.datasets import dump_svmlight_file

DATA_PATH = '/home/linghao/Question-Relevance-in-VQA/data/true_vs_false_premise/extended/processed/final_data/'
CHUNK_SIZE = 10000
DIM_FEAT = 400


def process_data(mode):
	print('Processing %s dataset ...' % mode)
	assert mode in ['train', 'val', 'test', 'qrpe_train', 'qrpe_test', 'secondorder_train', 'secondorder_test']
	start_time = time.time()
	
	f = open('{0}{1}_Xy.txt'.format(DATA_PATH, mode))
	Xy = []
	part = 0
	i = 0

	def save_chunk():
		nonlocal Xy
		nonlocal part
		nonlocal i

		Xy = np.array(Xy)
		X = Xy[:, :DIM_FEAT]
		y = Xy[:, -1]
		dump_svmlight_file(X, y, '{0}sl/{1}_Xy.sl.part{2}'.format(DATA_PATH, mode, part))

		Xy = []
		part += 1
		i = 0
		print('Chunk #%d saved, time elapsed = %.2fs' % (part-1, time.time()-start_time))

	for line in f:
		Xy.append(literal_eval(line.strip()))
		i += 1
		if i % CHUNK_SIZE == 0:
			save_chunk()

	if i != 0:
		save_chunk()

	f.close()

	f = open('{0}{1}_Xy.sl.all'.format(DATA_PATH, mode), 'w')

	for i in range(part):
		file = '{0}sl/{1}_Xy.sl.part{2}'.format(DATA_PATH, mode, i)
		g = open(file)
		for line in g:
			f.write(line)
		g.close()
		os.remove(file)
		print('Chunk %d/%d merged, time elapsed = %.2fs' % (i, part-1, time.time()-start_time))

	f.close()


if __name__ == '__main__':
	# process_data('train')
	# process_data('val')
	# process_data('test')
	# process_data('qrpe_train')
	# process_data('qrpe_test')
	process_data('secondorder_train')
	process_data('secondorder_test')
