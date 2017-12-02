"""Processes raw tuples and creates annotated question image pairs
"""

import sys
import json
import numpy as np

MAX_NEG_IMGS_PER_TUPLE = 10

def get_tuple(imid, qid, relevance, source):
    return '%s\t%s\t%d\t%d' % (imid, qid, relevance, source)

def main():
    input_filename, vqa_questions_filename, order = sys.argv[1:]
    order = int(order)
    assert order in [1, 2]
    
    with open(input_filename, 'r') as fp:
        data = json.load(fp)
    
    tuples = set()
    with open(vqa_questions_filename, 'r') as fp:
        vqa_questions = json.load(fp)['questions']

    for d in vqa_questions:
        tuples.add(get_tuple(d['image_id'], d['question_id'], True, True))

    for d in data:
        qid = d['qid']
        rel_id = d['rel_imid']
        irr_ids = d['irr_imids']
        tuples.add(get_tuple(rel_id, qid, True, True))
        for irr_id in irr_ids[:MAX_NEG_IMGS_PER_TUPLE]:
            tuples.add(get_tuple(irr_id, qid, False, order == 1))
    
    tuples = list(tuples)
    idxs = np.random.permutation(len(tuples))
    for i in idxs:
        print(tuples[i])

if __name__ == '__main__':
    main()
