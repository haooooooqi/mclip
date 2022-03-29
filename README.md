## ImageNet classification using ViT

### Introduction

This is a repo for reproducing a crash observed in Kaiming's code. The repo was based on the [official FLAX ImageNet example](https://github.com/google/flax/tree/main/examples/imagenet), with ViT model definition from the [official ViT repo](https://github.com/google-research/vision_transformer). EmaState is added to support Exponential Moving Average, which is what caused the crash.

Enviroment:
flax                         0.4.1
jax                          0.3.4
jaxlib                       0.3.2
tensorflow                   2.8.0

### Reproducing the crash

1. Request a TPU VM with v3-256 (the crash was only seen in v3-256).

1. Run the script `run_init_remote.sh` to set up the TPU VM. The repo is cloned into `flax_dev`. You may need a GitHub ID here, as the repo is private.

1. Run the script `run_staging.sh`.

1. The bug only happens with this setting in v3-256:
```
CONFIG=cfg_vit_large
batch=4096
ema=True
donate=True
```
It does not happen in other settings.



