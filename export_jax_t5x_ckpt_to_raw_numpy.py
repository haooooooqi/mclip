import argparse
import os
import numpy as np
from t5x.checkpoints import load_t5x_checkpoint


def flatten_dict(x, prefix="", out=None):
    if out is None:
        out = {}

    if isinstance(x, np.ndarray):
        assert prefix not in out
        out[prefix] = x
    else:
        assert isinstance(x, dict)
        for k, v in x.items():
            new_prefix = (prefix + "." + k) if prefix != "" else k
            flatten_dict(v, new_prefix, out)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jax_t5x_dir", type=str, required=True)
    parser.add_argument("--out_npz_file", type=str, required=True)
    args = parser.parse_args()

    states = load_t5x_checkpoint(args.jax_t5x_dir)
    states = flatten_dict(states)

    os.makedirs(os.path.dirname(args.out_npz_file), exist_ok=True)
    np.savez(args.out_npz_file, **states)
    print(f"saved JAX checkpoint to {args.out_npz_file}")


if __name__ == "__main__":
    main()
