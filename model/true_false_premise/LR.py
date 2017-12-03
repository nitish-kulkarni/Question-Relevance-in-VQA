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



def train(train_file, feat_dim=400, learning_rate=0.5, reg_coef=0.0001, verbose=False):
	start_time = time.time()
	weights = np.zeros(feat_dim)
	decay = 1 - 2 * learning_rate * reg_coef
	i = 0
	f = open(train_file)

	for line in f:
		vec = ast.literal_eval(line.strip())
		feat = np.array(vec[:feat_dim])
		label = vec[-1]

		score = np.dot(weights, feat)
		p = sigmoid(score)
		delta = feat * learning_rate * (label - p)
		weights = weights * decay + delta

		i += 1
		if verbose and i % 50000 == 0:
			print('%d lines processed, time elapsed = %.2fs' % (i, time.time()-start_time))

	return weights, time.time()-start_time


def test(test_file, weights):
	total = 0
	tp, tn, fp, fn = 0, 0, 0, 0
	f = open(test_file)

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

	return float(tp + tn) / total, (tp, tn, fp, fn)


if __name__ == '__main__':
	train_file = DATA_PATH + 'Xy_train.txt'
	test_file = DATA_PATH + 'Xy_test.txt'

	weights, time = train(train_file, verbose=True)
	np.save('./LR_model', weights)
	acc, stats = test(test_file, weights)

	print('Accuracy = %.4f, Time = %.2fs' % (acc, time))
	print(stats)
