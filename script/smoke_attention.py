#!/usr/bin/env python3
"""Run one real-data forward/backward pass for an attention selector config."""

import argparse
import inspect

import torch
from torch.utils.data import DataLoader

from dataset import TriDataset
from model.TRIPLEX.TRIPLEX import TRIPLEX
from utils.train_utils import load_config_with_default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/ST/andrew/TRIPLEX_attention_soft.yaml",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--selector-ckpt-path", default=None)
    args = parser.parse_args()

    cfg = load_config_with_default(args.config)
    cfg.DATA.mode = "cv"
    cfg.DATA.fold = args.fold
    cfg.MODEL.num_genes = cfg.DATA.num_outputs
    if args.selector_ckpt_path is not None:
        cfg.MODEL.selector_ckpt_path = args.selector_ckpt_path

    dataset_args = inspect.getfullargspec(TriDataset.__init__).args[1:]
    dataset = TriDataset(
        **{
            name: cfg.DATA[name]
            for name in dataset_args
            if name in cfg.DATA and name not in {"mode", "phase", "fold"}
        },
        mode="cv",
        phase="train",
        fold=args.fold,
    )
    batch = next(iter(DataLoader(dataset, batch_size=args.batch_size)))

    model_args = inspect.getfullargspec(TRIPLEX.__init__).args[1:]
    model = TRIPLEX(
        **{name: cfg.MODEL[name] for name in model_args if name in cfg.MODEL}
    ).cuda().train()

    for key, value in batch.items():
        if torch.is_tensor(value):
            batch[key] = value.cuda(non_blocking=True)
    for key in ("pid", "sid"):
        batch[key] = batch[key].reshape(-1)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        result = model(**batch, dataset=dataset, phase="train")
        loss = result["loss"]
    loss.backward()

    selector_grad = sum(
        parameter.grad.detach().float().abs().sum().item()
        for parameter in model.neighbor_selector.parameters()
        if parameter.grad is not None
    )
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss: {loss.item()}")
    if selector_grad <= 0:
        if cfg.MODEL.selector_mode != "frozen_hard":
            raise RuntimeError("No non-zero gradient reached the neighbor selector")

    if cfg.MODEL.selector_mode == "frozen_hard" and any(
        parameter.requires_grad for parameter in model.neighbor_selector.parameters()
    ):
        raise RuntimeError("Frozen selector unexpectedly has trainable parameters")

    print(
        "attention smoke test passed: "
        f"fold={args.fold}, batch={args.batch_size}, "
        f"loss={loss.item():.6f}, selector_grad_l1={selector_grad:.6f}"
    )


if __name__ == "__main__":
    main()
