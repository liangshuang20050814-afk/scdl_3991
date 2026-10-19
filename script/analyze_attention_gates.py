#!/usr/bin/env python3
"""Summarize learned soft gates and implied hard Top-k selections by fold."""

import argparse
from glob import glob
from pathlib import Path

import h5py
import pandas as pd
import torch

from model.TRIPLEX.selector import TargetConditionedSelector, load_selector_state
from utils.train_utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg = load_config(str(run_dir / "config.yaml"))
    rows = []

    for fold in range(cfg.TRAINING.num_k):
        checkpoint_paths = glob(str(run_dir / f"fold{fold}" / "*.ckpt"))
        checkpoint_paths = [p for p in checkpoint_paths if not p.endswith("last.ckpt")]
        if len(checkpoint_paths) != 1:
            raise RuntimeError(
                f"Expected one best checkpoint for fold {fold}, got {checkpoint_paths}"
            )

        selector = TargetConditionedSelector(
            emb_dim=cfg.MODEL.emb_dim,
            selector_dim=cfg.MODEL.selector_dim,
            num_selected_patches=cfg.MODEL.num_selected_patches,
            temperature=cfg.MODEL.selector_temperature,
            use_similarity_prior=cfg.MODEL.selector_use_similarity_prior,
            similarity_prior_weight=cfg.MODEL.selector_similarity_prior_weight,
            learned_score_weight=cfg.MODEL.selector_learned_score_weight,
        ).eval()
        load_selector_state(selector, checkpoint_paths[0])

        split = pd.read_csv(Path(cfg.DATA.data_dir) / "splits" / f"test_{fold}.csv")
        soft_values = []
        effective_counts = []
        hard_counts = []
        per_spot_ranges = []
        spot_count = 0

        with torch.no_grad():
            for slide_id in split["sample_id"]:
                emb_path = (
                    Path(cfg.DATA.data_dir)
                    / "emb" / "neighbor" / cfg.DATA.model_name
                    / f"{slide_id}.h5"
                )
                with h5py.File(emb_path, "r") as handle:
                    size = handle["embeddings"].shape[0]
                    for start in range(0, size, args.batch_size):
                        stop = min(start + args.batch_size, size)
                        embeddings = torch.from_numpy(handle["embeddings"][start:stop])
                        mask = torch.from_numpy(handle["mask_tb"][start:stop])
                        result = selector(embeddings, mask, "soft")
                        valid_context = mask.bool()
                        valid_context[:, mask.shape[1] // 2] = False
                        gates = result["soft_gate"]

                        soft_values.append(gates[valid_context])
                        effective_counts.append((gates * valid_context).sum(1))
                        hard_counts.append(result["hard_gate"].sum(1))
                        masked_max = gates.masked_fill(~valid_context, -torch.inf).max(1).values
                        masked_min = gates.masked_fill(~valid_context, torch.inf).min(1).values
                        has_context = valid_context.any(1)
                        per_spot_ranges.append((masked_max - masked_min)[has_context])
                        spot_count += stop - start

        soft_values = torch.cat(soft_values).float()
        effective_counts = torch.cat(effective_counts).float()
        hard_counts = torch.cat(hard_counts).float()
        per_spot_ranges = torch.cat(per_spot_ranges).float()
        rows.append({
            "fold": fold,
            "spots": spot_count,
            "soft_gate_mean": soft_values.mean().item(),
            "soft_gate_std": soft_values.std().item(),
            "soft_gate_min": soft_values.min().item(),
            "soft_gate_max": soft_values.max().item(),
            "effective_context_mean": effective_counts.mean().item(),
            "hard_total_mean": hard_counts.mean().item(),
            "within_spot_gate_range_mean": per_spot_ranges.mean().item(),
        })

    table = pd.DataFrame(rows)
    output = Path(args.output) if args.output else run_dir / "attention_gate_summary.csv"
    table.to_csv(output, index=False)
    print(table.to_string(index=False))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
