# DeLMA

DeLMA(Deep Learning Morph Analyzer) 혹은 NNN PosTagger(NHN Neural NLP PosTagger)은 [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor)에 기반을 둔 형태소 분석기입니다.
형태소 분석기 외에 새로운 Problem 모듈 추가는 `Tensor2Tensor` 프로젝트의 [new_problem.md](https://github.com/tensorflow/tensor2tensor/blob/master/docs/new_problem.md)을 참고하시기 바랍니다.

## 설치

프로젝트를 clone한 후 필수 패키지를 설치합니다. (python=3.6 권장)

```bash
git clone https://github.com/nhnent/DeLMA
# activate virtual environment (such as conda) if needed
cd DelMA
pip install -r requirement.txt
pip install tensorflow-gpu==1.10 #if you are not using gpu, then use `pip install tensorflow`
```

## 학습데이터 생성

학습데이터 생성은 원본파일을 학습에 사용될 수 있는 형태로 파일을 변형하는 단계입니다.
세부적으로 학습데이터 생성은 (1) train/dev/test로 파일을 분할해주는 과정과 (2) vocab 파일 생성하는 과정으로 이루어져 있습니다. 
(`datagen.sh` 참고)

```bash
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
```

## 학습

`tensor2tensor` 패키지를 활용하여 모델을 학습시킵니다.
(`train.sh` 참고)

```bash
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
WORKER_GPU=2
export CUDA_VISIBLE_DEVICES=1,2
t2t-trainer \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --worker_gpu=$WORKER_GPU \
  --output_dir=$TRAIN_DIR
```

## 디코딩

학습이 완료되면 학습된 모델을 로드하여 디코딩을 할 수 있습니다.
([decode.sh](https://github.com/nhnent/DeLMA/blob/dev/decode.sh) 참고)

```bash
#!/bin/sh
USR_DIR=pos_tagger
PROBLEM=pos_sejong800k
MODEL=transformer
HPARAMS=transformer_base

TRAIN_DIR=$HOME/t2t_train/$PROBLEM/$MODEL-$HPARAMS #학습된 모델이 저장된 경로
DATA_DIR=$HOME/t2t_data/$PROBLEM
DECODE_FROM_FILE=$HOME/t2t_decode/sejong_raw_refine_subword_input.txt #디코딩을 위한 입력 파일
DECODE_TO_FILE=$HOME/t2t_decode/sejong_raw_refine_subword_decoded.txt #디코딩 결과를 출력할 파일

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
 ```

## 성능평가

완료된 디코딩(혹은 prediction)에 대하여 정확도를 평가합니다.
디코딩된 토큰이 입력된 토큰과 일치하고 순서가 일치하는지에 대하여 평가합니다.
([test.sh](https://github.com/nhnent/DeLMA/blob/dev/test.sh) 참고)

```bash
#!/bin/sh

ANSWER_FILE=usr_dir/t2t_decode/sejong_raw_refine_tokenize_output.txt #정답파일
OUTPUT_FILE=usr_dir/t2t_decode/sejong_raw_refine_tokenize_decoded.txt #디코딩 출력파일

python -u pos_tagger_tester.py \
    --answer_file=$ANSWER_FILE \
    --output_file=$OUTPUT_FILE
```
