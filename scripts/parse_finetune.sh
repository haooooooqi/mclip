#!/bin/bash

LOG="$1"

if [[ -z "$LOG" ]]; then
    echo "Usage: $0 /path/to/log/file"
    exit 1
fi

if [ -f "$LOG" ]; then
    grep "Last accuracy" "$LOG" | cut -f7,10 -d' ' | tr -d ' '
fi
