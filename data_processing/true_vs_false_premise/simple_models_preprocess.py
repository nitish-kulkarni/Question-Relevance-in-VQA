import ast
import numpy as np
import sklearn
import subprocess

DATA_PATH = '/home/linghao/Question-Relevance-in-VQA/data/true_vs_false_premise/extended/processed/final_data/'
DIM_TXT_FEAT = 100


def process_data(mode):
    assert mode in ['train', 'val', 'test', 'qrpe_train', 'qrpe_test', 'secondorder_train', 'secondorder_test']
    
    f = open(DATA_PATH+'processed_%s_data.txt' % mode)
    g = open(DATA_PATH+'%s_questions.raw' % mode, 'w')
    labels = []
    i = 0
    for line in f:
        img_feat, question, label = line.strip().split('\t')
        g.write(question+'\n')
        label = int(label)
        labels.append(label)
        i += 1
    g.close()

    if mode == 'train':
        cmd = '/home/linghao/fastText-0.1.0/fasttext skipgram -dim %d -input %s -output %s' % \
                (DIM_TXT_FEAT, DATA_PATH+'train_questions.raw', DATA_PATH+'fastText_model')
        subprocess.call(cmd, shell=True)

    cmd = '/home/linghao/fastText-0.1.0/fasttext print-sentence-vectors %s < %s > %s' % \
            (DATA_PATH+'fastText_model.bin', DATA_PATH+('%s_questions.raw'%mode), DATA_PATH+('%s_questions.vec'%mode))
    subprocess.call(cmd, shell=True)

    n = len(labels)

    f = open(DATA_PATH+'processed_%s_data.txt' % mode)
    g = open(DATA_PATH+'%s_questions.vec' % mode)
    h = open(DATA_PATH+'%s_Xy.txt' % mode, 'w')

    for i in range(n):
        if i % 50000 == 0: print(i)
        line = f.readline()
        img_feat, _, __ = line.strip().split('\t')
        img_feat = ast.literal_eval('[%s]'%img_feat)
        line = g.readline()
        txt_feat = [float(x) for x in line.strip().split(' ')[-DIM_TXT_FEAT:]]
        vec = img_feat + txt_feat + [labels[i]]
        h.write(str(vec)+'\n')

    h.close()


if __name__ == '__main__':
	# process_data('train')
	# process_data('val')
	# process_data('test')
	# process_data('qrpe_train')
	# process_data('qrpe_test')
	process_data('secondorder_train')
	process_data('secondorder_test')
