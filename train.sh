#!/bin/sh
USR_DIR=pos_tagger
PROBLEM=pos_sejong800k_subword
MODEL=transformer
HPARAMS=transgiformer_base

DATA_DIR=usr_dir/t2t_data/$PROBLEM #학습된 데이터의 위치
TRAIN_DIR=usr_dir/t2t_train/$PROBLEM/$MODEL-$HPARAMS #학습된 모델이 저장될 위치

mkdir -p $TRAIN_DIR

# Train
# *  If you run out of memory, add --hparams='batch_size=1024'.
WORKER_GPU=2 #사용할 gpu 개수
export CUDA_VISIBLE_DEVICES=1,2 #사용할 gpu 번호 지정. $WORKER_GPU와 개수가 동일해야 함
t2t-trainer \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --worker_gpu=$WORKER_GPU \
  --output_dir=$TRAIN_DIR