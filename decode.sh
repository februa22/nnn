USR_DIR=pos_tagger
PROBLEM=pos_sejong800k
MODEL=transformer
HPARAMS=transformer_base

DATA_DIR=$HOME/www/nnn/usr_dir/t2t_data/$PROBLEM
TMP_DIR=$HOME/www/nnn/usr_dir/t2t_datagen
TRAIN_DIR=$HOME/www/nnn/usr_dir/t2t_train/$PROBLEM/$MODEL-$HPARAMS
WORKER_GPU=1

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# Generate data
# * The following files should be stored in $TMP_DIR
#   * Data: pos_sejong800k.pairs
#   * ELMo resources:
#     * Options: elmo/options.json
#     * Weights: elmo/weights.hdf5
t2t-datagen \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM
