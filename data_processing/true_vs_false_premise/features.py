"""Get question and image features from IDs
"""

import json
import torchfile
import numpy as np
from sklearn.decomposition import PCA

np.random.seed(42)

DATA_PATH = "../../../project_nongit/Data/"

# Question and image maps
QUESTIONS_MAP = json.load(open(DATA_PATH + '/vqa2_questions.json', 'r'))
# QUESTIONS_MAP = json.load(open(DATA_PATH + '/vqa1_questions.json', 'r'))
COCO_TRAIN_IMG_MAP = json.load(open("%s/coco_train_image_map.json" % DATA_PATH, "r"))
COCO_VAL_IMG_MAP = json.load(open("%s/coco_val_image_map.json" % DATA_PATH, "r"))
VG_IMG_MAP = json.load(open("%s/vg_image_map.json" % DATA_PATH, "r"))

### Collecting fc7 features for images
COCO_TRAIN_FEAT = torchfile.load("%s/train_fc7.t7" % DATA_PATH)
COCO_VAL_FEAT = torchfile.load("%s/val_fc7.t7" % DATA_PATH)
VG_FEAT = torchfile.load("%s/vg_fc7.t7" % DATA_PATH)

TRAIN_PATH = DATA_PATH + "train_firstorder_data.txt"
VAL_PATH = DATA_PATH + "val_firstorder_data.txt"

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

### Computing PCA features
uniq_img_ids = {}
with open(TRAIN_PATH, "r") as f:
    for line in f:
        line = line.strip("\n")
        arr = line.split("\t")
        uniq_img_ids[arr[0]] = arr[3]
with open(VAL_PATH, "r") as f:
    for line in f:
        line = line.strip("\n")
        arr = line.split("\t")
        uniq_img_ids[arr[0]] = arr[3]
image_ids = []
coco = []
for key in uniq_img_ids:
    image_ids.append(key)
    coco.append(uniq_img_ids[key])

img_features = get_image_features(image_ids, coco)
pca = PCA(n_components=300)
img_transformed_features = pca.fit_transform(img_features)

f = open(TRAIN_PATH, "r")
train_lines = f.readlines()
f.close()

### Creating train and validation data
train_data_split = 0.8
total_data_instances = len(train_lines)
perm = np.random.permutation(range(total_data_instances))
train_end_idx = int(train_data_split*total_data_instances)
train_data = [train_lines[i] for i in perm[0:train_end_idx]]
val_data = [train_lines[i] for i in perm[train_end_idx:total_data_instances]]

fid = open("processed_train_data.txt", "w")
for line in train_data:
    line = line.strip("\n")
    (img_id, qid, rel, src) = line.split("\t")
    img_fts = ",".join(["{:.4f}".format(x) for x in img_transformed_features[image_ids.index(img_id)]])
    ques = get_question(qid)
    fid.write(img_fts + "\t" + ques + "\t" + str(rel) + "\n")
fid.close()

fid = open("processed_val_data.txt", "w")
for line in val_data:
    line = line.strip("\n")
    (img_id, qid, rel, src) = line.split("\t")
    img_fts = ",".join(["{:.4f}".format(x) for x in img_transformed_features[image_ids.index(img_id)]])
    ques = get_question(qid)
    fid.write(img_fts + "\t" + ques + "\t" + str(rel) + "\n")
fid.close()

### Creating test data
f = open(VAL_PATH, "r")
test_lines = f.readlines()
f.close()

fid = open("processed_test_data.txt", "w")
for line in test_lines:
    line = line.strip("\n")
    (img_id, qid, rel, src) = line.split("\t")
    img_fts = ",".join(["{:.4f}".format(x) for x in img_transformed_features[image_ids.index(img_id)]])
    ques = get_question(qid)
    fid.write(img_fts + "\t" + ques + "\t" + str(rel) + "\n")
fid.close()







