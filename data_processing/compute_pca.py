import numpy as np
from sklearn.decomposition import PCA
import json
import ast

#Open json data path
data = json.load(open("../../project_nongit/Data/data.json", "r"))

img_ids = []
img_features = []
for obj in data:
	img_ids.append(obj["image_id"])
	img_features.append(np.array(ast.literal_eval(obj["image_features_fc7"])))

img_features = np.array(img_features)

pca = PCA(n_components=300)
img_transformed_features = pca.fit_transform(img_features)

i=0
for obj in data:
	img_id = img_ids[i]
	obj["image_features_fc7"] = str(list(img_transformed_features[i]))
	if img_id!=obj["image_id"]:
		print "Error!"
	i+=1

output_path = "../../project_nongit/Data"
with open('%s/data_transformed.json' % output_path, 'w') as fp:
	json.dump(data, fp)