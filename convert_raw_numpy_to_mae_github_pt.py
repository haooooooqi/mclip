import argparse
import os
import numpy as np
import torch
from collections import OrderedDict


def convert_jax_t5x_to_pt(t5x_np_ckpt, remove_k_bias=False):
    t5x_param_keys = [k for k in t5x_np_ckpt.keys() if not k.startswith("state.")]
    pt_ckpt = OrderedDict()

    def _replace(s1, s2, s3):
        assert s2 in s1
        return s1.replace(s2, s3)

    for k in t5x_param_keys:
        if ".encoderblock_" in k or ".decoderblock_" in k:
            # this is a ViT block
            block_idx = int(k.split(".")[2].split("_")[1])
            pt_k_prefix = ("blocks." if ".encoderblock_" in k else "decoder_blocks.") + f"{block_idx}."

            is_kv = (
                k.endswith(".key.kernel")
                or k.endswith(".key.bias")
                or k.endswith(".value.kernel")
                or k.endswith(".value.bias")
            )
            if is_kv:
                # the transformer attention's K and V weights will be handled when processing Q weights
                continue
            elif ".LayerNorm_0.scale" in k:
                pt_ckpt[pt_k_prefix + "norm1.weight"] = torch.from_numpy(t5x_np_ckpt[k])
            elif ".LayerNorm_0.bias" in k:
                pt_ckpt[pt_k_prefix + "norm1.bias"] = torch.from_numpy(t5x_np_ckpt[k])
            elif ".LayerNorm_1.scale" in k:
                pt_ckpt[pt_k_prefix + "norm2.weight"] = torch.from_numpy(t5x_np_ckpt[k])
            elif ".LayerNorm_1.bias" in k:
                pt_ckpt[pt_k_prefix + "norm2.bias"] = torch.from_numpy(t5x_np_ckpt[k])
            elif ".MlpBlock_0.Dense_0.kernel" in k:
                pt_ckpt[pt_k_prefix + "mlp.fc1.weight"] = torch.from_numpy(t5x_np_ckpt[k]).permute(1, 0)
            elif ".MlpBlock_0.Dense_0.bias" in k:
                pt_ckpt[pt_k_prefix + "mlp.fc1.bias"] = torch.from_numpy(t5x_np_ckpt[k])
            elif ".MlpBlock_0.Dense_1.kernel" in k:
                pt_ckpt[pt_k_prefix + "mlp.fc2.weight"] = torch.from_numpy(t5x_np_ckpt[k]).permute(1, 0)
            elif ".MlpBlock_0.Dense_1.bias" in k:
                pt_ckpt[pt_k_prefix + "mlp.fc2.bias"] = torch.from_numpy(t5x_np_ckpt[k])
            elif ".MultiHeadDotProductAttention_0.out.kernel" in k:
                pt_ckpt[pt_k_prefix + "attn.proj.weight"] = torch.from_numpy(t5x_np_ckpt[k]).permute(1, 0)
            elif ".MultiHeadDotProductAttention_0.out.bias" in k:
                pt_ckpt[pt_k_prefix + "attn.proj.bias"] = torch.from_numpy(t5x_np_ckpt[k])
            elif ".MultiHeadDotProductAttention_0.query.kernel" in k:
                pt_ckpt[pt_k_prefix + "attn.qkv.weight"] = torch.cat(
                    [
                        torch.from_numpy(t5x_np_ckpt[k]),
                        torch.from_numpy(t5x_np_ckpt[_replace(k, ".query.", ".key.")]),
                        torch.from_numpy(t5x_np_ckpt[_replace(k, ".query.", ".value.")]),
                    ],
                    dim=1,
                ).permute(1, 0)
            elif ".MultiHeadDotProductAttention_0.query.bias" in k:
                if remove_k_bias:
                    pt_ckpt[pt_k_prefix + "attn.q_bias"] = torch.from_numpy(t5x_np_ckpt[k])
                    pt_ckpt[pt_k_prefix + "attn.v_bias"] = torch.from_numpy(t5x_np_ckpt[_replace(k, ".query.", ".value.")])
                else:
                    pt_ckpt[pt_k_prefix + "attn.qkv.bias"] = torch.cat(
                        [
                            torch.from_numpy(t5x_np_ckpt[k]),
                            torch.from_numpy(t5x_np_ckpt[_replace(k, ".query.", ".key.")]),
                            torch.from_numpy(t5x_np_ckpt[_replace(k, ".query.", ".value.")]),
                        ],
                        dim=0,
                    )
            else:
                raise Exception(f"unknown T5X param key: {k}")
        elif k == "target.cls":
            pt_ckpt["cls_token"] = torch.from_numpy(t5x_np_ckpt[k])
        elif k == "target.posembed_encoder.pos_embedding":
            pt_ckpt["pos_embed"] = torch.from_numpy(t5x_np_ckpt[k])
        elif k == "target.embedding.kernel":
            pt_ckpt["patch_embed.proj.weight"] = torch.from_numpy(t5x_np_ckpt[k]).permute(3, 2, 0, 1)
        elif k == "target.embedding.bias":
            pt_ckpt["patch_embed.proj.bias"] = torch.from_numpy(t5x_np_ckpt[k])
        elif k == "target.Transformer.encoder_norm.scale":
            pt_ckpt["norm.weight"] = torch.from_numpy(t5x_np_ckpt[k])
        elif k == "target.Transformer.encoder_norm.bias":
            pt_ckpt["norm.bias"] = torch.from_numpy(t5x_np_ckpt[k])
        elif k == "target.mask_token":
            pt_ckpt["mask_token"] = torch.from_numpy(t5x_np_ckpt[k])
        elif k == "target.posembed_decoder.pos_embedding":
            pt_ckpt["decoder_pos_embed"] = torch.from_numpy(t5x_np_ckpt[k])
        elif k == "target.TransformerDecoder.decoder_norm.scale":
            pt_ckpt["decoder_norm.weight"] = torch.from_numpy(t5x_np_ckpt[k])
        elif k == "target.TransformerDecoder.decoder_norm.bias":
            pt_ckpt["decoder_norm.bias"] = torch.from_numpy(t5x_np_ckpt[k])
        elif k == "target.bottleneck.kernel":
            pt_ckpt["decoder_embed.weight"] = torch.from_numpy(t5x_np_ckpt[k]).permute(1, 0)
        elif k == "target.bottleneck.bias":
            pt_ckpt["decoder_embed.bias"] = torch.from_numpy(t5x_np_ckpt[k])
        elif k == "target.pred.kernel":
            pt_ckpt["decoder_pred.weight"] = torch.from_numpy(t5x_np_ckpt[k]).permute(1, 0)
        elif k == "target.pred.bias":
            pt_ckpt["decoder_pred.bias"] = torch.from_numpy(t5x_np_ckpt[k])
        else:
            raise Exception(f"unknown T5X param key: {k}")

    return pt_ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_npz_file", type=str, required=True)
    parser.add_argument("--out_mae_pt_file", type=str, required=True)
    parser.add_argument("--remove_k_bias", action="store_true")
    args = parser.parse_args()

    t5x_np_ckpt = np.load(args.raw_npz_file)
    out_pt_ckpt = convert_jax_t5x_to_pt(t5x_np_ckpt, remove_k_bias=args.remove_k_bias)
    os.makedirs(os.path.dirname(args.out_mae_pt_file), exist_ok=True)
    torch.save({"model": out_pt_ckpt}, args.out_mae_pt_file)
    print(f"saved MAE GitHub-fomat PyTorch checkpoint to {args.out_mae_pt_file}")


if __name__ == "__main__":
    main()
