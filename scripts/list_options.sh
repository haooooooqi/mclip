#!/bin/bash

INCLUDE=$1
EXCLUDE1=$2
EXCLUDE2=$3
EXCLUDE3=$4
EXCLUDE4=$5

if [ -z "$INCLUDE" ]; then
    INCLUDE="*"
fi

example_dir=default
echo "-------PRINTING FOLDER LIST-------"
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
    if [[ $dir = *"@"* ]]; then
        example_dir=$dir
        break
    fi
done
echo "----------------------------------"


echo "-------PRINTING OPTION LIST-------"
if [[ $example_dir = *"@"* ]]; then
    echo $example_dir | tr '%' '\n' | cut -f1 -d'@' | paste -s -d'\t'
fi
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
    if [[ $dir = *"@"* ]]; then
        echo $dir | tr '%' '\n' | cut -f2 -d'@' | paste -s -d'\t'
    fi
done
if [[ $example_dir = *"@"* ]]; then
    echo $example_dir | tr '%' '\n' | cut -f1 -d'@' | paste -s -d'\t'
fi
echo "----------------------------------"