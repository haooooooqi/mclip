import os
import torch
from collections import OrderedDict
from flax.training import checkpoints


def flatten_jax_state_tree(states):
    out = {}

    def _traverse_dict(d, prefix):
        for k, v in d.items():
            assert isinstance(k, str) and "." not in k
            new_prefix = k if prefix == "" else prefix + "." + k
            if isinstance(v, dict):
                _traverse_dict(v, new_prefix)
            else:
                out[new_prefix] = v

    _traverse_dict(states, "")

    return out


def convnext_jax2pt(jax_state_dict):
    """Convert a JAX ConvMAE/ConvNeXt state dict into a PyTorch state dict"""
    out_pt_state_dict = OrderedDict()

    for name_jax, var_jax in jax_state_dict.items():
        var_pt = torch.from_numpy(var_jax.copy())
        name_splits_jax = name_jax.split(".")
        name_splits_pt = name_splits_jax[:]  # copy JAX names

        # convert the layer names
        if name_splits_jax[0].startswith("downsample_layers"):
            layer_full_ids = name_splits_jax[0].replace("downsample_layers", "")
            n_stage = int(layer_full_ids[0])
            n_depth = int(layer_full_ids[1:])
            name_splits_pt[0] = f"downsample_layers.{n_stage}.{n_depth}"
        elif name_splits_jax[0].startswith("stages"):
            layer_full_ids = name_splits_jax[0].replace("stages", "")
            n_stage = int(layer_full_ids[0])
            n_depth = int(layer_full_ids[1:])
            name_splits_pt[0] = f"stages.{n_stage}.{n_depth}"
        elif name_splits_jax[0] in ["norm", "head"]:
            pass
        else:
            print(f"skipping param: {name_jax} with shape {var_pt.shape} since it's not part of ConvNeXt")
            continue

        # convert the weights names and tensor formats
        if name_splits_jax[-1] == "kernel":
            if var_pt.dim() == 4:
                # nn.Conv kernel -- JAX (H, W, Cin, Cout) => PT (Cout, Cin, H, W)
                var_pt = var_pt.permute(3, 2, 0, 1)
            elif var_pt.dim() == 2:
                # nn.Dense -- JAX (Din, Dout) => PT (Dout, Din)
                var_pt = var_pt.permute(1, 0)
            else:
                raise Exception(f"unknown kernel param: {name_jax} with shape {var_pt.shape}")
            name_splits_pt[-1] = "weight"
        elif name_splits_jax[-1] == "scale":
            # nn.LayerNorm
            assert var_pt.dim() == 1, f"unknown scale param: {name_jax} with shape {var_pt.shape}"
            name_splits_pt[-1] = "weight"

        name_pt = ".".join(name_splits_pt)
        out_pt_state_dict[name_pt] = var_pt

    return out_pt_state_dict


def convert_convnext_ckpt_jax2pt(input_jax_ckpt_path, output_pt_ckpt_path):
    """
    Convert a JAX ConvMAE/ConvNeXt checkpoint file to a PyTorch checkpoint file
    - input_jax_ckpt_path (str): input JAX checkpoint path (can be gs:// bucket)
    - output_pt_ckpt_path (str): output PyTorch checkpoint path (can be gs:// bucket)
    """
    assert input_jax_ckpt_path.startswith("gs://") or os.path.exists(input_jax_ckpt_path)

    jax_state_tree = checkpoints.restore_checkpoint(input_jax_ckpt_path, target=None)
    # keep only the parameters in ConvNext trunk
    jax_state_dict = flatten_jax_state_tree(jax_state_tree["params"]["ConvNeXt_0"])
    jax2pt_state_dict = convnext_jax2pt(jax_state_dict)
    pt_ckpt = {"model": jax2pt_state_dict}

    # save the output checkpoint (either to GCS buckets or filesystem)
    if output_pt_ckpt_path.startswith("gs://"):
        try:
            from google.cloud import storage
        except ImportError:
            print("please run `pip install google-cloud-storage` to install the GCS Python API")
            raise
        gcs_out_path_splits = output_pt_ckpt_path[len("gs://") :].split("/")
        bucket_name = gcs_out_path_splits[0]
        blob_name = "/".join(gcs_out_path_splits[1:])
        storage_client = storage.Client(bucket_name)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        with blob.open("wb", ignore_flush=True) as f:
            torch.save(pt_ckpt, f)
    else:
        os.makedirs(os.path.dirname(output_pt_ckpt_path), exist_ok=True)
        torch.save(pt_ckpt, output_pt_ckpt_path)
    print(f"PyTorch ConvNeXt checkpoint saved to {output_pt_ckpt_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    args = parser.parse_args()
    convert_convnext_ckpt_jax2pt(args.input, args.output)


if __name__ == "__main__":
    main()
