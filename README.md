## MAE fine-tuning in JAX

Written by Kaiming.

The MAE pre-training branch is `mae.torchloader` using PyTorch dataloader.

### Getting Started
- **Warning**: This repo is under development and not well documented yet
- Check https://github.com/google/flax/tree/main/examples/imagenet for ImageNet R50 training in JAX and TPU VM setup.
- Check https://github.com/google-research/vision_transformer for the official ViT code (in JAX).
- See `run_script.sh` for an example command line to debug in "local" TPU VM (v3-8).
- See `run_remote.sh` for an example script to run in "remote" TPU VMs (like v3-256).
