"""Example script to get features from data
"""

import sys
import features

# with open('../../data/true_vs_false_premise/qrpe/train_data.txt', 'r') as fp:
with open('../../data/true_vs_false_premise/extended/processed/train_firstorder_data.txt', 'r') as fp:
    for line in fp:
        image_id, qid, relevance, coco = line.strip().split('\t')
        feat = get_image_feature(image_id, int(coco) == 1)
        question = get_question(qid)
        y = int(relevance)
