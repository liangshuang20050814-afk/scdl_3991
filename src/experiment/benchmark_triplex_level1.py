"""Benchmark mask versus compact neighbor tokens on a real TRIPLEX batch."""

import argparse
import csv
from glob import glob
from pathlib import Path
import re

import torch

from dataset import DataInterface
from model import ModelInterface
from utils.train_utils import load_config, fix_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def best_checkpoint(log_dir, fold):
    paths = [
        path for path in glob(str(log_dir / f"fold{fold}" / "*.ckpt"))
        if Path(path).name != "last.ckpt"
    ]
    if not paths:
        raise FileNotFoundError(f"No checkpoint found in {log_dir}/fold{fold}")

    def epoch(path):
        match = re.search(r"epoch=(\d+)", Path(path).name)
        return int(match.group(1)) if match else -1

    return max(paths, key=epoch)


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def measure(model, batch, training, warmup, iterations, device):
    model.train(training)

    def step():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            result = model(**batch, phase="train")
            loss = result["loss"]
        if training:
            loss.backward()
            model.zero_grad(set_to_none=True)

    for _ in range(warmup):
        if training:
            step()
        else:
            with torch.inference_mode():
                step()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    memory_before = torch.cuda.memory_allocated(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        if training:
            step()
        else:
            with torch.inference_mode():
                step()
    end.record()
    torch.cuda.synchronize(device)
    elapsed_ms = start.elapsed_time(end)
    peak_memory = torch.cuda.max_memory_allocated(device)
    batch_size = batch["img"].shape[0]
    return {
        "milliseconds_per_step": elapsed_ms / iterations,
        "spots_per_second": batch_size * iterations / (elapsed_ms / 1000),
        "peak_allocated_mib": peak_memory / (1024 ** 2),
        "incremental_peak_mib": (peak_memory - memory_before) / (1024 ** 2),
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    device = torch.device("cuda")
    log_dir = Path("logs") / args.config_name / args.timestamp
    cfg = load_config(log_dir / "config.yaml")
    cfg.config = args.config_name
    cfg.DATA.mode = "cv"
    cfg.DATA.fold = args.fold
    cfg.DATA.train_dataloader.shuffle = False
    cfg.MODEL.num_genes = cfg.DATA.num_outputs
    cfg.MODEL.gene_path = (
        f"{cfg.DATA.data_dir}/{cfg.DATA.gene_type}_{cfg.DATA.num_genes}genes.json"
    )
    checkpoint = best_checkpoint(log_dir, args.fold)
    fix_seed(cfg.GENERAL.seed)

    dm = DataInterface(dataset_name=cfg.DATA.dataset_name, data_config=cfg.DATA)
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    batch = move_batch(batch, device)
    batch["pid"] = batch["pid"].view(-1)
    batch["sid"] = batch["sid"].view(-1)
    batch["dataset"] = dm.train_dataset

    rows = []
    for compact in (False, True):
        cfg.MODEL.compact_neighbor_tokens = compact
        interface = ModelInterface.load_from_checkpoint(
            checkpoint,
            model_name=cfg.MODEL.model_name,
            config=cfg,
        ).to(device)
        model = interface.model
        for training in (False, True):
            for repeat in range(1, args.repeats + 1):
                result = measure(
                    model,
                    batch,
                    training,
                    args.warmup,
                    args.iterations,
                    device,
                )
                row = {
                    "config": args.config_name,
                    "timestamp": args.timestamp,
                    "fold": args.fold,
                    "checkpoint": checkpoint,
                    "phase": "training" if training else "forward",
                    "implementation": "compact" if compact else "mask",
                    "budget": cfg.DATA.num_selected_patches,
                    "batch_size": batch["img"].shape[0],
                    "repeat": repeat,
                    **result,
                }
                rows.append(row)
                print(row, flush=True)
        del interface, model
        torch.cuda.empty_cache()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
