import argparse

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("first")
    parser.add_argument("second")
    args = parser.parse_args()

    first = torch.load(args.first, map_location="cpu")
    second = torch.load(args.second, map_location="cpu")
    if first.keys() != second.keys():
        raise SystemExit("state_dict keys differ")

    for key in first:
        left = first[key]
        right = second[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise SystemExit(f"tensor metadata differs: {key}")
        if not torch.equal(left, right):
            difference = (left.float() - right.float()).abs().max().item()
            raise SystemExit(f"tensor values differ: {key}, max_abs={difference}")

    print(f"bitwise_equal tensors={len(first)}")


if __name__ == "__main__":
    main()
