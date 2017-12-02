import torchfile
import json
import numpy as np

### Collecting necessary image ids for extracting features
unique_image_ids = {}
base_path = '/Users/nitish/acads/10605/project/data/true_vs_false_premise'
data_train = json.load(open("%s/train.json" % base_path,"r"))
data_test = json.load(open("%s/test.json" % base_path,"r"))
n_train = len(data_train)
n_test = len(data_test)

### Collecting fc7 features for images
base_path = '/Volumes/Nitish-Passport/10605_project_files/data_generation/premise/generate_IQ_pairs/img_data'
coco_train_feat = torchfile.load("%s/train_fc7.t7" % base_path)
coco_val_feat = torchfile.load("%s/val_fc7.t7" % base_path)
vg_feat = torchfile.load("%s/vg_fc7.t7" % base_path)
coco_train_dict = json.load(open("%s/coco_train_dict.json" % base_path, "r"))
coco_val_dict = json.load(open("%s/coco_val_dict.json" % base_path, "r"))
vg_dict = json.load(open("%s/vg_dict.json" % base_path, "r"))

# Data output path
output_path = '/Users/nitish/acads/10605/project/data/true_vs_false_premise'

def get_fc7(imid):
	if imid in coco_train_dict:
		return coco_train_feat[coco_train_dict[imid]]

	if imid in coco_val_dict:
		return coco_train_feat[coco_val_dict[imid]]

	if imid in vg_dict:
		return vg_feat[vg_dict[imid]]

	raise 'Unknown image id ', imid

def get_feature(imid, question, relevance):
    return {
        'question': question,
        'image_id': imid,
        'image_features_fc7': str(list(get_fc7(str(imid)))),
        'relevance': relevance
    }

data = []
for d in data_train + data_test:
    for val in d['tuplist']:
        data.append(get_feature(val['rel_imid'], d['q'], True))
        data.append(get_feature(val['irr_imid'], d['q'], False))

np.random.shuffle(data)

print 'Saving %d Q-I pairs' % len(data)
with open('%s/data.json' % output_path, 'w') as fp:
    json.dump(data, fp)

