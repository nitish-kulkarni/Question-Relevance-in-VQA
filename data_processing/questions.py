import sys

# Usage:
# Visual questions
# cat filename | python questions.py V
# N for NonVisual questions

for line in sys.stdin:
    tag, question = line.strip().split('\t')
    if tag == sys.argv[1]:
        print(question)

