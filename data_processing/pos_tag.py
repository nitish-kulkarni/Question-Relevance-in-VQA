import spacy
import sys

nlp = spacy.load('en')

def postags(question):
    return [(str(token.text), str(token.pos_)) for token in nlp(question.decode('utf-8'))]

def main():
	for line in sys.stdin:
		line = line.strip("\n")
		label, question = line.split("\t")
		output_string = ''
		for word_pos in postags(question):
			output_string += '%s|%s|' % word_pos
		sys.stdout.write(label + "\t" + output_string[:-1].encode('utf-8').strip() + "\n")

if __name__ == '__main__':
	main()