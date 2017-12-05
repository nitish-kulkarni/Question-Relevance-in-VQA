import numpy as np
import time
import ast

DATA_PATH = '/home/linghao/Question-Relevance-in-VQA/data/true_vs_false_premise/extended/processed/final_data/'
OVERFLOW = 20


def sigmoid(score):
	score = min(score, OVERFLOW)
	score = max(score, -OVERFLOW)
	exp = np.exp(score)
	return exp / (1 + exp)


def train(train_file, val_file, name='first order train dataset', feat_dim=400, learning_rate=0.5, reg_coef=1e-4, max_iters=1, verbose=False):
	if verbose:
		print('Training ...')
	start_time = time.time()
	weights = np.zeros(feat_dim)
	decay = 1 - 2 * learning_rate * reg_coef

	k = 0
	while True:
		f = open(train_file)
		i = 0
		total = 0
		tp, tn, fp, fn = 0, 0, 0, 0

		for line in f:
			vec = ast.literal_eval(line.strip())
			feat = np.array(vec[:feat_dim])
			label = vec[-1]

			score = np.dot(weights, feat)
			p = sigmoid(score)

			total += 1
			if (label == 1 and p >= 0.5): tp += 1
			if (label == 0 and p < 0.5): tn += 1
			if (label == 0 and p >= 0.5): fp += 1
			if (label == 1 and p < 0.5): fn += 1

			delta = feat * learning_rate * (label - p)
			weights = weights * decay + delta

			i += 1
			if verbose and i % 10000 == 0:
				print('%d lines processed, time elapsed = %.2fs' % (i, time.time()-start_time))

		if verbose:
			report(name, tp, tn, fp, fn)

		val_acc = test(val_file, weights)
		if verbose:
			print('Epoch #%d, validation accuracy = %.2f\n' % (k+1, val_acc))

		np.save(DATA_PATH+('LR_%d.model'%k), weights)

		k += 1
		if k == max_iters:
			break
		decay = 1 - 2 * learning_rate * reg_coef / k / k

	avg_time = (time.time()-start_time) / max_iters
	if verbose:
		print('=== Average training time per epoch = %.2fs ===\n' % avg_time)

	return weights, val_acc


def test(test_file, weights, name=None, feat_dim=400, verbose=False):
	if verbose:
		print('Testing ...')
	start_time = time.time()
	total = 0
	tp, tn, fp, fn = 0, 0, 0, 0
	f = open(test_file)
	i = 0

	for line in f:
		vec = ast.literal_eval(line.strip())
		feat = np.array(vec[:feat_dim])
		label = vec[-1]

		score = np.dot(weights, feat)
		p = sigmoid(score)

		total += 1
		if (label == 1 and p >= 0.5): tp += 1
		if (label == 0 and p < 0.5): tn += 1
		if (label == 0 and p >= 0.5): fp += 1
		if (label == 1 and p < 0.5): fn += 1

		i += 1
		if verbose and i % 10000 == 0:
			print('%d lines processed, time elapsed = %.2fs' % (i, time.time()-start_time))

	if name is not None:
		report(name, tp, tn, fp, fn)
	return (tp + tn) / total


def report(name, tp, tn, fp, fn):
	print('=== Metrics on %s ===\n' % name)

	total = tp + tn + fp + fn
	accuracy = (tp + tn) / total
	precision_true = tp / (tp + fp)
	precision_false = tn / (tn + fn)
	recall_true = tp / (tp + fn)
	recall_false = tn / (tn + fp)
	fscore_true = (precision_true + recall_true) / 2
	fscore_false = (precision_false + recall_false) / 2
	support_true = tp + fn
	support_false = tn + fp

	print('Classification Report')
	headers = ['precision', 'recall', 'f1-score','support']
	head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers) + '\n'
	ret = head_fmt.format(u'', *headers, width=8)
	row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'
	ret += row_fmt.format('False', precision_false, recall_false, fscore_false, support_false, width=8, digits=2)
	ret += row_fmt.format('True', precision_true, recall_true, fscore_true, support_true, width=8, digits=2)
	print(ret)
	print('Accuracy = %.2f' % accuracy)

	print('Confusion Matrix')
	print('%d\t\t%d\n%d\t\t%d\n' % (tn, fp, fn, tp))


def param_search():
	train_file = DATA_PATH+'qrpe_Xy_train.txt'
	val_file = DATA_PATH+'qrpe_Xy_test.txt'

	for learning_rate in [0.5]:
		for reg_coef in [1e-6, 1e-5, 1e-4]:
			weights, acc = train(train_file, val_file, learning_rate=learning_rate, reg_coef=reg_coef, verbose=True)
			np.save(DATA_PATH+('qrpe_LR_lr_%f_reg_%f.model'%(learning_rate, reg_coef)), weights)
			print('lr = %f, reg = %f, acc = %f' % (learning_rate, reg_coef, acc))


def main():
	train_file = DATA_PATH+'Xy_train.txt'
	val_file = DATA_PATH+'Xy_val.txt'
	test_file = DATA_PATH+'Xy_test.txt'
	qrpe_train_file = DATA_PATH+'qrpe_Xy_train.txt'
	qrpe_test_file = DATA_PATH+'qrpe_Xy_test.txt'
	
	# weights = train(train_file=train_file, val_file=val_file, max_iters=5, verbose=True)
	
	# weights = np.load(DATA_PATH+'LR.model.npy')
	# test(val_file, weights, name='first order val dataset', verbose=True)
	# test(test_file, weights, name='first order test dataset', verbose=True)
	# test(qrpe_train_file, weights, name='qrpe train dataset', verbose=True)
	# test(qrpe_test_file, weights, name='qrpe test dataset', verbose=True)


if __name__ == '__main__':
	param_search()
	# main()
