"""Processes raw tuples and creates annotated question image pairs
"""

import sys
import json
import numpy as np

def get_tuple(imid, qid, relevance, source):
    return '%s\t%s\t%d\t%d' % (imid, qid, relevance, source)

def main():
    input_filename, order = sys.argv[1:]
    order = int(order)
    assert order in [1, 2]
    
    with open(input_filename, 'r') as fp:
        data = json.load(fp)
    
    tuples = set()
    for d in data:
        qid = d['qid']
        for tup in d['tuplist']:
            rel_id = tup['rel_imid']
            irr_id = tup['irr_imid']
            tuples.add(get_tuple(str(rel_id), str(qid), True, True))
            tuples.add(get_tuple(str(irr_id), str(qid), False, len(tup['irr_tuple']) == 0))

    tuples = list(tuples)
    idxs = np.random.permutation(len(tuples))
    for i in idxs:
        print(tuples[i])

if __name__ == '__main__':
    main()
