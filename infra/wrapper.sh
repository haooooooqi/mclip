#!/bin/bash

SCRIPT_NAME=$1
JOB_NAME=$2
NUM_CORE=$3
array=( $@ )
len=${#array[@]}

EXTRA_ARGS=${array[@]:2:$len}

LOG_DIR=${HOME}/outputs
if [ ! -d ${LOG_DIR} ]; then
    mkdir -p $LOG_DIR
fi

LOG_PREFIX="${LOG_DIR}/`date +'%Y-%m-%d_%H-%M-%S'`_$$_${JOB_NAME}"

SUBMITTED=0
if [ $NUM_CORE -eq 128 ]; then
    for job in xinleic-mae-0 xinleic-mae-1 xinleic-mae-2 xinleic-mae-3; do
        has_job=`~/mask/list.sh | grep $job`
        if [ -z "${has_job}" ]; then
            nohup $HOME/mask/${SCRIPT_NAME}.sh $JOB_NAME $job ${EXTRA_ARGS} 1>${LOG_PREFIX}.out 2>${LOG_PREFIX}.err &
            SUBMITTED=1
            break
        fi
    done
elif [ $NUM_CORE -eq 256 ]; then
    for job in xinleic-mae-ii-0 xinleic-mae-ii-1 xinleic-mae-ii-2 xinleic-mae-ii-3; do
        has_job=`~/mask/list.sh | grep $job`
        if [ -z "${has_job}" ]; then
            nohup $HOME/mask/${SCRIPT_NAME}.sh $JOB_NAME $job ${EXTRA_ARGS} 1>${LOG_PREFIX}.out 2>${LOG_PREFIX}.err &
            SUBMITTED=1
            break
        fi
    done
elif [ $NUM_CORE -eq 512 ]; then
    for job in xinleic-mae-iv-2 xinleic-mae-iv-1 xinleic-mae-iv-3 xinleic-mae-iv-0; do
        has_job=`~/mask/list.sh | grep $job`
        if [ -z "${has_job}" ]; then
            nohup $HOME/mask/${SCRIPT_NAME}.sh $JOB_NAME $job ${EXTRA_ARGS} 1>${LOG_PREFIX}.out 2>${LOG_PREFIX}.err &
            SUBMITTED=1
            break
        fi
    done
fi
