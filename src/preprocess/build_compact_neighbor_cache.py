"""Materialize static neighbor selections as physically compact HDF5 files."""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from dataset.tri_dataset import TriDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--method", choices=("random", "similarity", "diversity"), required=True
    )
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--selection-seed", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_selector(method, budget, seed):
    selector = object.__new__(TriDataset)
    selector.neighbor_selection = method
    selector.num_selected_patches = budget
    selector.selection_seed = seed
    return selector


def selected_indices(selected_mask, budget):
    counts = selected_mask.sum(dim=1)
    if torch.any(counts == 0) or torch.any(counts > budget):
        raise ValueError("Invalid number of selected tokens")
    rows, original_tokens = selected_mask.shape
    original_indices = torch.arange(original_tokens).expand(rows, -1)
    ordered = torch.where(selected_mask.bool(), original_indices, original_tokens)
    ordered = ordered.sort(dim=1).values[:, :budget]
    compact_mask = torch.arange(budget)[None, :] < counts[:, None]
    return ordered.masked_fill(~compact_mask, 0), compact_mask


def materialize_slide(source, destination, selector, budget, metadata):
    with h5py.File(source, "r") as handle:
        key = "embeddings" if "embeddings" in handle else "features"
        embeddings = np.asarray(handle[key][:])
        masks = np.asarray(handle["mask_tb"][:])

    tensor_embeddings = torch.from_numpy(embeddings)
    tensor_masks = torch.from_numpy(masks)
    selection = selector.select_neighbor_mask(
        tensor_embeddings,
        tensor_masks,
        source.stem,
        torch.arange(len(embeddings)),
    )
    indices, compact_mask = selected_indices(selection, budget)
    compact = tensor_embeddings.gather(
        1,
        indices.unsqueeze(-1).expand(-1, -1, tensor_embeddings.shape[-1]),
    )
    compact = compact.masked_fill(~compact_mask.unsqueeze(-1), 0)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".h5.tmp")
    with h5py.File(temporary, "w") as handle:
        handle.create_dataset("embeddings", data=compact.numpy())
        handle.create_dataset("mask_tb", data=compact_mask.numpy().astype(np.uint8))
        handle.create_dataset("token_indices", data=indices.numpy().astype(np.int16))
        for key, value in metadata.items():
            handle.attrs[key] = value
    os.replace(temporary, destination)


def validate_slide(source, destination, selector, budget):
    with h5py.File(source, "r") as original, h5py.File(destination, "r") as compact:
        key = "embeddings" if "embeddings" in original else "features"
        original_embeddings = original[key][:]
        original_masks = original["mask_tb"][:]
        cached_embeddings = compact["embeddings"][:]
        cached_masks = compact["mask_tb"][:].astype(bool)
        cached_indices = compact["token_indices"][:]

    expected_mask = selector.select_neighbor_mask(
        torch.from_numpy(original_embeddings),
        torch.from_numpy(original_masks),
        source.stem,
        torch.arange(len(original_embeddings)),
    )
    expected_indices, expected_compact_mask = selected_indices(expected_mask, budget)
    expected = torch.from_numpy(original_embeddings).gather(
        1,
        expected_indices.unsqueeze(-1).expand(-1, -1, original_embeddings.shape[-1]),
    )
    expected = expected.masked_fill(~expected_compact_mask.unsqueeze(-1), 0)
    if not np.array_equal(cached_indices, expected_indices.numpy()):
        raise ValueError(f"Token-index mismatch in {destination}")
    if not np.array_equal(cached_masks, expected_compact_mask.numpy()):
        raise ValueError(f"Mask mismatch in {destination}")
    if not np.array_equal(cached_embeddings, expected.numpy()):
        raise ValueError(f"Embedding mismatch in {destination}")


def main():
    args = parse_args()
    if not 1 <= args.budget <= 25:
        raise ValueError("budget must be between 1 and 25")
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    sources = sorted(input_dir.glob("*.h5"))
    if not sources:
        raise FileNotFoundError(f"No HDF5 files found in {input_dir}")

    selector = make_selector(args.method, args.budget, args.selection_seed)
    metadata = {
        "format": "triplex_compact_neighbor_v1",
        "method": args.method,
        "budget": args.budget,
        "selection_seed": args.selection_seed,
        "source_dir": str(input_dir),
    }
    for number, source in enumerate(sources, 1):
        destination = output_dir / source.name
        if destination.exists() and not args.overwrite:
            print(f"[{number}/{len(sources)}] reusing {destination.name}", flush=True)
        else:
            print(f"[{number}/{len(sources)}] creating {destination.name}", flush=True)
            materialize_slide(source, destination, selector, args.budget, metadata)
        validate_slide(source, destination, selector, args.budget)

    manifest = {
        **metadata,
        "slides": len(sources),
        "files": [source.name for source in sources],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Validated compact cache: {output_dir}")


if __name__ == "__main__":
    main()
