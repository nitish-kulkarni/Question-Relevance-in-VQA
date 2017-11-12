"""
Convert VQA source questions to SPICE format
"""
import simplejson as json
import copy
import sys

# SPICE requires a certain format to execute. Convert into format and retain image_id and question_id
def vqa_to_spice(ipfile, opfile):
	with open(ipfile, 'r') as ifile:
		vqa_data = json.load(ifile)
	vqa_questions = vqa_data['questions']
	q_spice = []
	q_object = {}
	for i in range(len(vqa_questions)):
		q_object['image_id'] = str(vqa_questions[i]['image_id']) + ' ' + str( vqa_questions[i]['question_id']) + ' ' + vqa_questions[i]['question']
		q_object['test']  = vqa_questions[i]['question']
		q_object['refs'] = ['']
		q_spice.append(copy.deepcopy(q_object))
	with open(opfile, 'w') as ofile:
		json.dump(q_spice, ofile)

def main():
    args = sys.argv[1:]
    vqa_to_spice(args[0], args[1])

if __name__ == '__main__':
   main()