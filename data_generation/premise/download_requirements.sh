#!/usr/bin/env bash

# Download required packages
wget https://nlp.stanford.edu/software/stanford-postagger-2017-06-09.zip -P qgen/
wget https://www.nodebox.net/code/data/media/linguistics.zip -P qgen/

# Download SPICE
wget http://www.panderson.me/images/SPICE-1.0.zip -P premises/

# Unzip SPICE
tar -xzf premises/SPICE-1.0.zip -C premises/

# Download Stanford models for SPICE
sh ./premises/SPICE-1.0/get_stanford_models.sh

# Download VQA questions
wget http://visualqa.org/data/mscoco/vqa/Questions_Train_mscoco.zip -P premises/
wget http://visualqa.org/data/mscoco/vqa/Questions_Val_mscoco.zip -P premises/

# Unzip VQA questions
tar -xzf premises/Questions_Train_mscoco.zip -C premises/
tar -xzf premises/Questions_Val_mscoco.zip -C premises/

# Unzip downloaded packages
tar -xzf qgen/stanford-postagger-2017-06-09.zip -C qgen/
tar -xzf qgen/linguistics.zip -C qgen/

# Rename folder
mkdir "qgen/stanfordnlp"
mv "qgen/stanford-postagger-2017-06-09" "qgen/stanford-nlp/stanford-postagger"

# Install python packages
pip install nltk simplejson