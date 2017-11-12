"""Prints to stdout the top n dissimar questions
for every question in stdin
Similarity is measured from word2vec similarity
of nouns, adjectives and verbs in the question
"""

import sys
from gensim.models.keyedvectors import KeyedVectors

from data_processing.pos_tag import postags

GOOGLE_WORD2VEC = 'data/word2vec_vectors/GoogleNews-vectors-negative300.bin'

def pretrained_model(filename):
    return KeyedVectors.load_word2vec_format(filename, binary=True)

def visual_questions():
    questions = []
    for line in sys.stdin:
        tag, question = line.strip().split('\t')
        if tag == 'V':
            questions.append(question)
    return questions

def question_words(question):
    """Identifier words for a question
    In this case, proper and common nouns
    """
    return [word for word, tag in postags(question) if tag in ['NOUN', 'PROPN', 'VERB'] ]

def dissimilar_questions(words_all_questions, words_question, top_n):
    """To be implemented
    """
    top_questions = []
    return top_questions

def main():
    model = pretrained_model(GOOGLE_WORD2VEC)
    questions = visual_questions()
    words_all_questions = [question_words(question) for question in questions]
    for i, question in enumerate(questions):
        print('%s\t%s' % (question, '|'.join(dissimilar_questions(words_all_questions[i], words_all_questions, 10))))

if __name__ == '__main__':
    main()
