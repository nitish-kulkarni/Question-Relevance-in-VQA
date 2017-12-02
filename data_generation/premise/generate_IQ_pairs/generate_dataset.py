import json

# Input the VQA question types file
with open("store/mscoco_question_types.txt", "r") as f:
    types = f.read().splitlines()
    types.sort(lambda x, y: cmp(len(x), len(y)), reverse=True)

# 'none of the above' added
types.append('none of the above')

# Function modified to return appropriate VQA question type
def determine_ques_type(question):
    # Return the answer and question types for the given question
    global types
    for ts in types:
        question = ' '.join(question.split())
        tsq = ts.split(",")[0]
        if question.lower().startswith(tsq):
            return ts.split(",")[1]

    return 'none of the above'

def to_filter(q):
	qtype = determine_ques_type(q) 
	return qtype  == "yes/no" or qtype == "number"

def parse(data):
	qid2ixlist = {}
	for ix, val in enumerate(data):
		if val["qid"] in qid2ixlist:
			qid2ixlist[val["qid"]].append(ix)
		else:
			qid2ixlist[val["qid"]] = [ix]
	parsed = []
	for k,v in qid2ixlist.iteritems():

		if to_filter(data[v[0]]["q"]):
			continue
		
		tuplist, tuples = [], []
		for pos in v:
			tupobj = data[pos]
			tuplist.append({
				"rel_tuple": tupobj["rel_tuple"],
				"irr_tuple": tupobj["irr_tuple"],
				"rel_imid": tupobj["rel_imid"],
				"irr_imid": tupobj["irr_imid"]
				})

			tuples.append(tupobj["rel_tuple"])

		parsed.append({
			"q": data[v[0]]["q"],
			"qid": data[v[0]]["qid"],
			"tuplist": tuplist,
			"tuples": tuples
			})
	return parsed

def create_splits(data):
	ix = int(len(data)*2/3)
	return data[:ix], data[ix:]

def main():
	with open("../train/tups/objects_train_tup.json") as f1, open("../train/tups/attributes_train_tup.json") as f2, \
		 open("../test/tups/objects_test_tup.json") as f3, open("../test/tups/attributes_test_tup.json") as f4:
		otrain = json.load(f1)
		otest = json.load(f3)

		atrain = json.load(f2)
		atest = json.load(f4)
		
		objects_train, objects_test = parse(otrain), \
	 							   	  parse(otest),
		
		attributes_train, attributes_test = parse(atrain), \
	 							   	  		parse(atest),
	
		train = parse(otrain + atrain)
		test = parse(otest + atest)

		# Hack: Just adding SPICE tuples to everything
		test_spice = test
		objects_test_spice = objects_test
		attributes_test_spice = attributes_test

	with open("../train/objects_train.json", "w") as f1, open("../train/attributes_train.json", "w") as f2, \
		 open("../test/objects_test.json", "w") as f3, open("../test/attributes_test.json", "w") as f4, \
		 open("../train/train.json", "w") as f5, open("../test/test.json", "w") as f6, \
		 open("../test/test_w_spice.json", "w") as f7, open("../test/objects_test_w_spice.json", "w") as f9, \
		 open("../test/attributes_test_w_spice.json", "w") as f9:
		
		json.dump(objects_train, f1, indent=4)
		json.dump(attributes_train, f2, indent=4)

		json.dump(objects_test, f3, indent=4)
		json.dump(attributes_test, f4, indent=4)
		
		json.dump(train, f5, indent=4)
		json.dump(test, f6, indent=4)
		
		json.dump(test_spice, f7, indent=4)
		json.dump(objects_test_spice, f8, indent=4)
		json.dump(attributes_test_spice, f9, indent=4)


if __name__ == "__main__":
	main()