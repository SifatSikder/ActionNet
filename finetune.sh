#!/bin/bash
set -e

# Where the pre-trained Inception Resnet V2 checkpoint is saved to.
#PRETRAINED_CHECKPOINT_DIR=/home/cheer/video_test/classifier/partial_clip/checkpoints

# Where the pre-trained Inception Resnet V2 checkpoint is saved to.

#EVALUATIONSET_NAME=python_0_0

# Where the training (fine-tuned) checkpoint and logs will be saved to.
MODEL_NAME=action_vgg_e
TRAIN_DIR=/content/drive/MyDrive/IIT/4-2/SPL-3/Model

# Where the dataset is saved to.
TRAINSET_DIR=/content/drive/MyDrive/IIT/4-2/SPL-3/Model
#EVALUATIONSET_DIR=/home/cheer/video_test/corre/data/${EVALUATIONSET_NAME}

#exponential   fixed

python3 train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_split_name=train \
  --dataset_dir=${TRAINSET_DIR} \
  --model_name=${MODEL_NAME} \
  --max_number_of_steps=10000 \
  --batch_size=8 \
  --learning_rate=0.01 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=600 \
  --save_summaries_secs=300 \
  --log_every_n_steps=10 \
  --optimizer=adam \
  --weight_decay=0.00004 \
  --input_mode=0 \
  --output_mode=0
