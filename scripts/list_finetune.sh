#!/bin/bash

PHRASE=$2
INCLUDE=$3
EXCLUDE1=$4
EXCLUDE2=$5
EXCLUDE3=$6
EXCLUDE4=$7

if [ "$1" = T ]; then
    ~/mae_jax/scripts/list_options.sh "$3" "$4" "$5" "$6" "$7"
fi

if [ -z "$PHRASE" ]; then
    PHRASE=finetune
fi

if [ -z "$INCLUDE" ]; then
    INCLUDE="*"
fi

echo "-------PRINTING RESULT LIST-------"
for dir in $INCLUDE; do
    if [[ $dir = *old-* ]]; then
        continue
    fi
    if [ ! -z "${EXCLUDE1}" ]; then
        if [[ $dir = *${EXCLUDE1}* ]]; then
            continue
        fi
    fi
    if [ ! -z "${EXCLUDE2}" ]; then
        if [[ $dir = *${EXCLUDE2}* ]]; then
            continue
        fi
    fi
    if [ ! -z "${EXCLUDE3}" ]; then
        if [[ $dir = *${EXCLUDE3}* ]]; then
            continue
        fi
    fi
    if [ ! -z "${EXCLUDE4}" ]; then
        if [[ $dir = *${EXCLUDE4}* ]]; then
            continue
        fi
    fi
    echo $dir
    has_res=false
    for i in "$dir"/${PHRASE}/${PHRASE}_main_*.log; do
        if [ ! -f $i ]; then
            continue
        fi
        res=`~/vit_jax/scripts/parse_finetune.sh $i`;
        if [ ! -z "$res" ]; then
            echo $res;
            has_res=true
        fi
    done
    if [ "$has_res" = false ]; then
        echo ""
    fi
done
echo "----------------------------------"

echo "-------PRINTING RESULT ONLY-------"
for dir in $INCLUDE; do
    if [[ $dir = *old-* ]]; then
        continue
    fi
    if [ ! -z "${EXCLUDE1}" ]; then
        if [[ $dir = *${EXCLUDE1}* ]]; then
            continue
        fi
    fi
    if [ ! -z "${EXCLUDE2}" ]; then
        if [[ $dir = *${EXCLUDE2}* ]]; then
            continue
        fi
    fi
    if [ ! -z "${EXCLUDE3}" ]; then
        if [[ $dir = *${EXCLUDE3}* ]]; then
            continue
        fi
    fi
    if [ ! -z "${EXCLUDE4}" ]; then
        if [[ $dir = *${EXCLUDE4}* ]]; then
            continue
        fi
    fi

    has_res=false
    for i in "$dir"/${PHRASE}/${PHRASE}_main_*.log; do
        if [ ! -f $i ]; then
            continue
        fi
        res=`~/vit_jax/scripts/parse_finetune.sh $i`;
        if [ ! -z "$res" ]; then
            echo $res;
            has_res=true
        fi
    done
    if [ "$has_res" = false ]; then
        echo ""
    fi
done
echo "----------------------------------"