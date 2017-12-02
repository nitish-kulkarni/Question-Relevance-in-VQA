Dataset Format for {train, val}_{first, second}order_data.txt

    Each line has tab separated values:
        v1  v2  v3  v4\n
    
    where,
        v1: image id
        v2: question id
        v3: question-relevance, 0 or 1
        v4: image data source, 0 for Visual Genome and 1 for COCO

To extract features from data, use functions in data_processing/true_vs_false_premise/features.py

Steps:
    * Add all the necessary json and t7 files in a folder, and set DATA_PATH in features.py as the path to the folder
    * Set QUESTIONS_MAP to vqa1 map or vqa2 map based on whether the dataset used is qrpe or extended dataset respectively
    * Refer to data_processing/true_vs_false_premise/test_features.py for feature extraction
