import os
import sys
import json

""" 
Clean up artifacts from SPICE
"""
question_items = ["what", "where", "who", "why", "does"]
nn_items = ["type", "kind", "color", "picture", "photo", "image"]
adjective_items = ["visible"]
tofilter_items = question_items + nn_items + adjective_items

def is_clean(tuple):
	for item in tuple:
		if item in tofilter_items:
			return False
	return True

if len(sys.argv):
	
	print sys.argv

	input_file = sys.argv[1]
	output_file = sys.argv[2]

	with open(input_file) as ip, open(output_file, "w") as op:
		data = json.loads(ip.read())
		results = []
		for i in xrange(0, len(data)):	
			proposition = data[i]
			test_tuples = proposition["test_tuples"]
			result = {}
			prop_data = proposition["image_id"].split()
			result["question"] = " ".join(prop_data[2::])
			result["question_id"] = prop_data[1]
			result["image_id"] = prop_data[0]
			result["tuples"] = []
			for test_tuple in test_tuples:
				if is_clean(test_tuple["tuple"]):
					result["tuples"].append(test_tuple["tuple"])
			results.append(result)
			
		json.dump(results, op)
