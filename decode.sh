#!/bin/sh
USR_DIR=pos_tagger
PROBLEM=pos_sejong800k
MODEL=transformer
HPARAMS=transformer_base

TRAIN_DIR=$HOME/t2t_train/$PROBLEM/$MODEL-$HPARAMS
DATA_DIR=$HOME/t2t_data/$PROBLEM
DECODE_FROM_FILE=$HOME/t2t_decode/sejong_raw_refine_subword_input.txt
DECODE_TO_FILE=$HOME/t2t_decode/sejong_raw_refine_subword_decoded.txt

BEAM_SIZE=4
ALPHA=0.6

t2t-decoder \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --decode_from_file=$DECODE_FROM_FILE \
  --decode_to_file=$DECODE_TO_FILE