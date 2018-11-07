#!/bin/sh

ANSWER_FILE=usr_dir/t2t_decode/sejong_raw_refine_tokenize_output.txt
OUTPUT_FILE=usr_dir/t2t_decode/sejong_raw_refine_tokenize_decoded.txt

python -u pos_tagger/pos_tagger_tester.py \
    --answer_file=$ANSWER_FILE \
    --output_file=$OUTPUT_FILE