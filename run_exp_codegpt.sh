#!/bin/bash

LANG="powershell"
DATADIR="/content/drive/MyDrive/tesi_magistrale/dataset/json"
OUTPUTDIR="/content/model"
PRETRAINDIR="microsoft/CodeGPT-small-py"  # will download pre-trained CodeGPT model
LOGFILE="text2code.log"
NUM_EPOCHS=1
NUM_TRAIN_SAMPLES=901
BATCH_SIZE=4
STEPS=$(($NUM_EPOCHS * $NUM_TRAIN_SAMPLES / $BATCH_SIZE))
SAVE_STEPS=$(($STEPS / 5))

echo $STEPS $SAVE_STEPS $BATCH_SIZE

python $PWD/Text-Code/text-to-code/code/run.py \
        --data_dir=$DATADIR \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --model_type=gpt2 \
        --do_train \
        --do_infer \
        --node_index 0 \
        --gpu_per_node 1 \
        --learning_rate=5e-5 \
        --weight_decay=0.01 \
        --evaluate_during_training \
        --per_gpu_train_batch_size=1 \
        --per_gpu_eval_batch_size=4 \
        --gradient_accumulation_steps=$BATCH_SIZE \
        --num_train_epochs=$NUM_EPOCHS \
        --logging_steps=10 \
        --save_steps=$SAVE_STEPS \
        --save_total_limit=1 \
        --overwrite_output_dir \
        --log_file=$LOGFILE \
        --seed=42