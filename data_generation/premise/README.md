# Implemented using the methodology in paper:

"The Promise of Premise: Harnessing Question Premises in Visual Question Answering (https://arxiv.org/abs/1705.00601)"

## Requirements

The download requrements are the file: download requirements.sh

To install:
chmod +x download_requirements.sh
./download_requirements

## For relevance prediction and explanation models

VGG 16 image features for MS COCO train, val and Visual Genome images are required for relevance explanation and prediction models. They can be downloaded from [here](https://filebox.ece.vt.edu/~aroma/web/img_data/).

## Steps to extract premise tuples

python vqa_to_spice.py OpenEnded_mscoco_{train2014, val2014}_questions.json spice_input.json

java -Xmx8G -jar SPICE-1.0/spice-*.jar spice_input.json -out spice_output.json -detailed

python process_output.py spice_output.json vqa_oe_tuples_filtered.json

## Steps to generate Image-Question pairs from extracted tuples



