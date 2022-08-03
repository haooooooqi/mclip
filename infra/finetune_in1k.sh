#!/bin/bash

JOB_NAME=$1
TPU_NAME=$2
NUM_CORE=$3
YAML_FT=$4
LOG_DIR=$5
MODEL_DIR=$6
TPU_EVAL_NAME=$7
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:7:$len}

STORAGE_BUCKET=gs://xinleic
DATA_DIR=${STORAGE_BUCKET}/in1k
echo 'TPU name: '${TPU_NAME}

for lr in .1; do
    for wd in 0.; do
        for cls in True; do
            ################################
            # NEED CHANGE HERE
            AFFIX=BY#${YAML_FT}_LR#${lr}_WD#${wd}_CLS#${cls}
            ################################
            FINETUNE_DIR=${MODEL_DIR}/${AFFIX}

            echo 'checkpoint dir: '${MODEL_DIR}
            echo 'finetune dir: '${FINETUNE_DIR}

            now=`date +'%Y-%m-%d_%H-%M-%S'`
            ################################
            # NEED CHANGE HERE
            python3 -u main.py \
                    ${EXTRA_ARGS} \
                    --salt=${JOB_NAME} \
                    --tpu=${TPU_NAME} \
                    --data_dir=${DATA_DIR} \
                    --model_dir=${FINETUNE_DIR} \
                    --checkpoint_dir=${MODEL_DIR} \
                    --mode=train \
                    --config_file=configs/tune/${YAML_FT}.yaml \
                    --num_cores=${NUM_CORE} \
                    --use_blur=False \
                    --use_color_jit=False \
                    --base_learning_rate=$lr \
                    --weight_decay=$wd \
                    --train_epochs=90 \
                    --train_batch_size=32768 \
                    --label_smoothing=0.1 \
                    --skip_host_call=True \
                    --is_unsup=False \
                    --optimizer='lars' \
                    --precision=float32 \
                    --use_last_bn=True \
                    --vit_sup_clstoken=${cls} \
                    --dry_run=False \
            2>&1 | tee $LOG_DIR/finetune_${AFFIX}_${now}.log
            ################################

            if [ ${PIPESTATUS[0]} -eq 0 ]; then
                bash tpu_up.sh $TPU_EVAL_NAME v2-8
                python3 -u main.py \
                        ${EXTRA_ARGS} \
                        --salt=${JOB_NAME} \
                        --tpu=$TPU_EVAL_NAME \
                        --data_dir=${DATA_DIR} \
                        --model_dir=${FINETUNE_DIR} \
                        --mode=eval \
                        --config_file=configs/tune/${YAML_FT}.yaml \
                        --num_cores=8 \
                        --eval_batch_size=16 \
                        --is_unsup=False \
                        --precision=float32 \
                        --use_last_bn=True \
                        --vit_sup_clstoken=${cls} \
                        --dry_run=False \
                2>&1 | tee $LOG_DIR/eval_${AFFIX}_${now}.log
                bash tpu_down.sh $TPU_EVAL_NAME
            fi
        done
    done 
done

# touch $LOG_DIR/ft_bn-done.flag