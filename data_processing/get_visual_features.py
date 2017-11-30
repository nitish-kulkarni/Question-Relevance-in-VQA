import torchfile
### To install torchfile, do pip install torchfile
import json

### Collecting necessary image ids for extracting features
unique_image_ids = {}
data_train = json.load(open("../../project_nongit/Data/train.json","r"))
data_test = json.load(open("../../project_nongit/Data/test.json","r"))
n_train = len(data_train)
n_test = len(data_test)

for i in range(n_train):
	n_id_pairs = len(data_train[i]["tuplist"])
	for j in range(n_id_pairs):
		unique_image_ids[data_train[i]["tuplist"][j]["irr_imid"]] = 1
		unique_image_ids[data_train[i]["tuplist"][j]["rel_imid"]] = 1
for i in range(n_test):
	n_id_pairs = len(data_test[i]["tuplist"])
	for j in range(n_id_pairs):
		unique_image_ids[data_test[i]["tuplist"][j]["irr_imid"]] = 1
		unique_image_ids[data_test[i]["tuplist"][j]["rel_imid"]] = 1

print "Number of unique image ids in all of train and test", len(unique_image_ids)

coco_train_feat = torchfile.load("../../project_nongit/Data/train_fc7.t7")
coco_val_feat = torchfile.load("../../project_nongit/Data/val_fc7.t7")
coco_train_dict = json.load(open("../../project_nongit/Data/coco_train_dict.json", "r"))
coco_val_dict = json.load(open("../../project_nongit/Data/coco_val_dict.json", "r"))

no_imgfeat_ids = {}
for img_id in unique_image_ids:
	if str(img_id) in coco_train_dict:
		unique_image_ids[img_id] = coco_train_feat[coco_train_dict[str(img_id)]]
	elif str(img_id) in coco_val_dict:
		unique_image_ids[img_id] = coco_val_feat[coco_val_dict[str(img_id)]]
	else:
		no_imgfeat_ids[img_id] = 1