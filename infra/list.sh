#!/bin/bash

echo "TPU jobs:"
ps aux | grep "google-cloud-sdk" | grep "python3"