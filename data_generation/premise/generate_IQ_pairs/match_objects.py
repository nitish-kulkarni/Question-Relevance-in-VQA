import sys
import json
import cPickle
import torchfile
import numpy as np
from tqdm import *
from nltk.corpus import wordnet as wn
from annoy import AnnoyIndex

sys.path.insert(0, './coco/PythonAPI')

from pycocotools.coco import COCO

N_TREES = 10

def strip_imgname(imgname):
	return int(imgname.split("_")[-1].split(".")[0])

def build_indexes(split):
	'''
	Build mapping from objects to image id's.
	'''
	print "Building Index..."
	
	annFile = "store/annotations/instances_" + split + "2014.json"
	coco=COCO(annFile)

	cats = coco.loadCats(coco.getCatIds())
	cococategories=[cat['name'] for cat in cats] 

	cat2img, img2cat = {}, {}

	for ix, cat in enumerate(cococategories):
		catIds = coco.getCatIds(catNms=[cat])
		cat2img[cat] = coco.getImgIds(catIds=catIds)

	print "Building NN index..."

	cocofeats_pca = np.load("store/vqa" + split + "feats_pca.npy") 		# COCO features from VGG-19 after PCA
	cocolist = torchfile.load("store/" + split + "_fc7_image_id.t7")	# Image ID mapping

	index = AnnoyIndex(100, metric="euclidean")
	imid2ix, ix2imid = {}, {}
	for ix, imid in enumerate(cocolist):
		imid2ix[imid] = ix
		ix2imid[ix] = imid
		ann = coco.loadAnns(coco.getAnnIds(imgIds=[strip_imgname(imid)]))
		img2cat[imid] = [coco.loadCats([a["category_id"]])[0]["name"] for a in ann]
		index.add_item(ix, cocofeats_pca[ix])

	index.build(N_TREES)
	index.save("store/obj_" + split + "index.ann")

	print "All indexes built."
	return cat2img, img2cat, cococategories, cocofeats_pca, index, imid2ix, ix2imid

def build_imgname(imid, phase):
	return "COCO_" + phase + "2014_" + str(imid).zfill(12) + ".jpg"

def retrieve_neg_img_list_coco(exclude, source_imid, cat2img, img2cat, cococategories, \
							   cocofeats_pca, index, imid2ix, ix2imid, split):
	'''
	Return negative image list, finding a negative image for each object in list
	that is an fc7 nearest-neighbor using COCO Annotations
	'''
	neg_img = False
	if exclude not in cococategories:
		return False

	key = build_imgname(source_imid, split)
	source_fc7 = cocofeats_pca[imid2ix[key]]

	neighbors = index.get_nns_by_vector(source_fc7, 1000)
	neighbor_imids = [ix2imid[neighbor] for neighbor in neighbors]
	for ix, neighbor in enumerate(neighbor_imids):
		key = int(neighbor.split("_")[-1].split(".")[0])
		cats = img2cat[neighbor]
		if exclude not in cats:
			return key

	return False

def extract_coco_objects(q, cococategories):
	return [cat for cat in cococategories if cat in q]

def main():
	'''
	Find negative object images for each tuple
	'''
	split = "train"
	cat2img, img2cat, cococategories, cocofeats_pca, index, imid2ix, ix2imid = build_indexes(split)
	out, seen = [], []
	input_file_name = "vqa_" + split + "_obj_tups"
	with open("input/" + input_file_name + ".json", "r") as f:
		data = json.loads(f.read())
		for i in tqdm(range(len(data))):
			qobj = data[i]
			if qobj["qid"] in seen:
				continue
			objlist = extract_coco_objects(qobj["q"], cococategories)
			seen.append(qobj["qid"])
			for obj in objlist:
				neg_img = retrieve_neg_img_list_coco(obj, qobj["imid"], cat2img, img2cat, cococategories, \
													 cocofeats_pca, index, imid2ix, ix2imid, split)

				if neg_img and (qobj["imid"] != neg_img):
					out.append({
						"q": qobj["q"],
						"qid": qobj["qid"],
						"rel_imid": qobj["imid"],
						"rel_tuple": obj,
						"irr_tuple": "",
						"irr_imid": neg_img
					})
	dest_folder = "train" if split == "train" else "test"			
	with open("../" + dest_folder + "/tups/objects_" + dest_folder + "_tup.json", "w") as f:
		json.dump(out, f, indent=4)

if __name__ == "__main__":					
	main()