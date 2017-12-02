import sys

n_rel = 0
n_irr = 0
for line in sys.stdin:
    imid, qid, relevance, source = line.strip().split('\t')
    n_rel += int(relevance)
    n_irr += 1 - int(relevance)

print('Total Relevant: %d' % n_rel)
print('Total Non-Relevant: %d' % n_irr)
print('Total IQ Pairs: %d' % (n_rel + n_irr))