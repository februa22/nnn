#!/bin/sh
USR_DIR=pos_tagger #사용자 정의 추가모듈파일 위치
PROBLEM=pos_sejong800k_subword #사용자 정의로 추가한 세종태그셋 problem. token(음절)방식을 사용할 경우 pos_sejong800k_token을 사용해야 함
MODEL=transformer #transformer 기본 모델 사용
HPARAMS=transformer_base #transformer의 기본 하이퍼파라미터 사용

TMP_DIR=usr_dir/t2t_datagen #원본파일의 위치.
# TMP_DIR 폴더 내에 pos_sejong800k_subword.pairs 혹은 pos_sejong800k_token.pairs이라는 이름으로 학습파일이 존재애햐 함
DATA_DIR=usr_dir/t2t_data/$PROBLEM #변형된 학습파일이 저장될 위치

mkdir -p $DATA_DIR

#학습데이터 생성
t2t-datagen \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM