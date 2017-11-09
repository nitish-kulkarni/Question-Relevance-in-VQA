import spacy
import sys

nlp = spacy.load('en')

for line in sys.stdin:
	line = line.strip("\n")
	(label,question) = line.split("\t")
	parsed_question = nlp(question.decode('utf-8'))
	output_string = ""
	for token in parsed_question:
		output_string += token.text + "|" + token.pos_ + "|"
	sys.stdout.write(label + "\t" + output_string[:-1].encode('utf-8').strip() + "\n")