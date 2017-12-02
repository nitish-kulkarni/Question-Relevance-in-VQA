"""Get question and image features from IDs
"""

import json
import torchfile
import numpy as np

DATA_PATH = '../../data/true_vs_false_premise/extended/processed'

# Question and image maps
QUESTIONS_MAP = json.load(open(DATA_PATH + '/vqa2_questions.json', 'r'))
# QUESTIONS_MAP = json.load(open(DATA_PATH + '/vqa1_questions.json', 'r'))
COCO_TRAIN_IMG_MAP = json.load(open("%s/coco_train_image_map.json" % DATA_PATH, "r"))
COCO_VAL_IMG_MAP = json.load(open("%s/coco_val_image_map.json" % DATA_PATH, "r"))
VG_IMG_MAP = json.load(open("%s/vg_image_map.json" % DATA_PATH, "r"))

### Collecting fc7 features for images
DATA_PATH = '/Volumes/Nitish-Passport/10605_project_files/data_generation/premise/generate_IQ_pairs/img_data'
COCO_TRAIN_FEAT = torchfile.load("%s/train_fc7.t7" % DATA_PATH)
COCO_VAL_FEAT = torchfile.load("%s/val_fc7.t7" % DATA_PATH)
VG_FEAT = torchfile.load("%s/vg_fc7.t7" % DATA_PATH)

def get_image_feature(image_id, coco=True):
    """inputs: 
        image_id: image id as string
        coco: boolean, feature from coco if True, else from Visual Genome
    outputs: numpy array of features
    """
    if coco:
        return COCO_TRAIN_FEAT[COCO_TRAIN_IMG_MAP[image_id]] if image_id in COCO_TRAIN_IMG_MAP else COCO_VAL_FEAT[COCO_VAL_IMG_MAP[image_id]]
    else:
        return VG_FEAT[VG_IMG_MAP[image_id]]

def get_question(question_id):
    """inputs: 
        question_id: vqa question id as string
    outputs: Corresponding VQA question as string
    """
    return QUESTIONS_MAP[question_id]

def get_image_features(image_ids, coco):
    """inputs: 
        image_id: list of image ids as strings
        coco: list of corresponding booleans,
                feature from coco if True, else from Visual Genome
    outputs: 2D numpy array of features
    """
    return np.array([get_image_feature(image_id, c) for image_id, c in zip(image_ids, coco)])

def get_questions(question_ids):
    """inputs: 
        question_id: vqa question ids as list of strings
    outputs: Corresponding VQA questions as list of strings
    """
    return [get_question(qid) for qid in question_ids]

