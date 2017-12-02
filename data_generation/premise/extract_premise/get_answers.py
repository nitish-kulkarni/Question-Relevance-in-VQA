"""Extract answers from train and val VQA annotation files
"""

import sys
import json
import tqdm

def main():
    input_filename, output_filename = sys.argv[1:]
    with open(input_filename, 'r') as fp:
        annotations = json.load(fp)['annotations']
    answers = {}
    for annotation in annotations:
        image_id = annotation['image_id']
        question_id = annotation['question_id']
        key = '%d %d' % (image_id, question_id)
        if key in answers:
            raise 'Unexpected'
        answers[key] = [ans['answer'] for ans in annotation['answers']]
    with open(output_filename, 'w') as fp:
        json.dump(answers, fp)

if __name__ == '__main__':
    main()
