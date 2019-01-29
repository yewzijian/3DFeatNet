#/bin/bash

DATASET_DIR=../data/oxford
LOG_DIR=./ckpt
GPU_ID=0

# Pretrain
python train.py \
  --data_dir $DATASET_DIR \
  --log_dir $LOG_DIR/pretrain \
  --augmentation Jitter RotateSmall Shift \
  --noattention --noregress \
  --num_epochs 2 \
  --gpu $GPU_ID

# Second stage training: Performance should saturate in ~60 epochs
python train.py \
  --data_dir $DATASET_DIR \
  --log_dir $LOG_DIR/secondstage \
  --checkpoint $LOG_DIR/pretrain/ckpt \
  --restore_exclude detection \
  --augmentation Jitter RotateSmall Shift Rotate1D \
  --num_epochs 70 \
  --gpu $GPU_ID