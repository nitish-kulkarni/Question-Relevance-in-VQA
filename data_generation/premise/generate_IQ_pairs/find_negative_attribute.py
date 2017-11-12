"""
This scripts generates second order data points 
for the QRPE dataset. It needs MS COCO and
Visual Genome Image data stored in ../../modes/img_data
Please check the specified folder to find how to
get this data.
"""
import sys
import os
import os.path
import multiprocessing
import json
import pickle
import torchfile
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

vgDir = "store/scene_graphs.json"
cocoDir = "store/all_tuples.json"
storedIndex = "store/attr_idx.json"
storedIndex_coco = "store/attr_idx_coco.json"
coco_train_path = '../../models/img_data/train_fc7.t7'
coco_val_path = '../../models/img_data/val_fc7.t7'
vg_path = '../../models/img_data/vg_fc7.t7'

with open('store/antonyms-and-others.pkl','r') as afile:
	[antonyms, colors, locations] = pickle.load(afile)

with open('../../models/img_data/coco_train_dict.json','r') as cfile:
	coco_train_dict = json.load(cfile)
with open('../../models/img_data/coco_val_dict.json','r') as cfile:
	coco_val_dict = json.load(cfile)
with open('../../models/img_data/vg_dict.json','r') as cfile:
	vg_dict = json.load(cfile)
coco_train_feat = torchfile.load(coco_train_path)
coco_val_feat = torchfile.load(coco_val_path)
vg_feat = torchfile.load(vg_path)

def build_indexes():
	if os.path.isfile(storedIndex):
		with open(storedIndex, "r") as index_file:
			attr_index = json.load(index_file)
		print "VG Index loaded .."
	else:
		print "Index not found, building Index..."
		attr_index = {}
		pool = multiprocessing.Pool(16)
		with open(vgDir,"r") as sgfile:
			scene_graphs = json.load(sgfile)
		for ix, scene_graph in enumerate(scene_graphs):
			if ix%10 == 0:
				print ix
			for sg_att_ob in scene_graph["attributes"]:
				sg_att = sg_att_ob["attribute"]
				if "attributes" in sg_att:
					for att in sg_att["attributes"]:
						key = sg_att["names"][0]
						obj = {
							"image_id": sg_att_ob['image_id'],
							"attr": att 
						}
						if key in attr_index:
							l = len(attr_index[key])
							if attr_index[key][l-1] != obj:
								attr_index[key].append(obj)
						else:
							attr_index[key] = [obj]
				else:
					pass
		print "Index built"
		with open(storedIndex, "w") as f:
		 	json.dump(attr_index, f)

	return attr_index

def build_indexes_coco():
	if os.path.isfile(storedIndex_coco):
		with open(storedIndex_coco, "r") as index_file:
			attr_index = json.load(index_file)
		print "COCO Index loaded .."
	else:
		print "Index not found, building Index..."
		attr_index = {}
		pool = multiprocessing.Pool(16)
		with open(cocoDir,"r") as cfile:
			coco_tuples = json.load(cfile)
		ix = 0
		print len(coco_tuples)
		for t in coco_tuples:
			if ix%10 == 0:
				print ix
			ix = ix + 1
			key = t['tuple'][0]
			obj = {
					"image_id": t['image_id'],
					"attr": t['tuple'][1] 
					}
			if key in attr_index:
				l = len(attr_index[key])
				attr_index[key].append(obj)
			else:
				attr_index[key] = [obj]
				
		print "Index built"
		with open(storedIndex_coco, "w") as f:
		 	json.dump(attr_index, f)

	return attr_index


def GetSimilarity(q, sg_obj, split='train'):
	if split == 'val':
		coco_feat = coco_val_feat
		coco_dict = coco_val_dict
	else:
		coco_feat = coco_train_feat
		coco_dict = coco_train_dict
	pred = q['tuple'][1]
	target_pred = sg_obj["attr"]
	if pred != target_pred:
		if (pred in colors and target_pred in colors)\
		or (pred in antonyms and target_pred in antonyms[pred]['antonym']):
			input_feat = coco_feat[coco_dict[str(q['image_id'])]]
			input_feat_sparse = sparse.csr_matrix(input_feat)
			target_feat = vg_feat[vg_dict[str(sg_obj['image_id'])]]
			target_feat_sparse = sparse.csr_matrix(target_feat)
			sim = cosine_similarity(input_feat_sparse,target_feat_sparse)[0][0]
			if sim > 0.999:
				sim = 0.0
			return(sim)
		else:
			return(0.0)
	else:
		return(0.0)
	
def build_dataset(input_file, split='train'):
	attr_index = build_indexes()
	out = []
	with open(input_file, "r") as f:
		data = json.load(f)
	for q in data:
		target_key = q["tuple"][0]
		#q_nb = True if (q["question"].split(' ')[0].lower() not in ['is','are','can','does','would','could','do']) else False
		if target_key in attr_index and q["tuple"] != ['dog','hot'] and\
		target_key not in ['object','one','thing','corner', 'enough','there','it','object','piece', 'section','letter' \
		'that','item','word','time','part','end','structure','level','name','stuff','body'] \
		and q["answer"].lower() not in ['no', 'none', 'noone', 'zero','0']  and target_key not in locations:
			target_pred = q["tuple"][1]
			scene_graph_objs = attr_index[target_key]
			curr_max, max_id = -1, -1	
			for scene_graph_obj in scene_graph_objs:
				sim = GetSimilarity(q,scene_graph_obj, split)
				if (sim > curr_max) and (float(sim) < 0.999):
					irr_obj = {
							"q": q["question"],
							"q_id": q["question_id"],
							"rel_imid": q["image_id"],
							"irr_imid": scene_graph_obj["image_id"],
							"rel_tuple": "_".join(q["tuple"]),
							"irr_tuple": q["tuple"][0]+"_"+scene_graph_obj["attr"],
							"sim": float(sim)
							}
					curr_max = sim
			if curr_max > 0.0:
				out.append(irr_obj)		
	out.sort(key=lambda x: x["sim"], reverse=False)
	return out

def main():
	args = sys.argv[1:]
	out = build_dataset(args[0], args[2])
	with open(args[1], "w") as f:
		json.dump(out, f, indent=4)

if __name__ == "__main__":					
	main()
