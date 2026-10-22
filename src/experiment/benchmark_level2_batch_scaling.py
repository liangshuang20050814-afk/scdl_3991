"""Measure Level 1/2 training throughput as batch size increases."""

import argparse
import copy
import csv
from glob import glob
from pathlib import Path
import re

import torch

from dataset import DataInterface
from model import ModelInterface
from utils.train_utils import fix_seed, load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--compact-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def best_checkpoint(log_dir, fold):
    paths = [
        path for path in glob(str(log_dir / f"fold{fold}" / "*.ckpt"))
        if Path(path).name != "last.ckpt"
    ]
    if not paths:
        raise FileNotFoundError(f"No checkpoint in {log_dir}/fold{fold}")
    return max(
        paths,
        key=lambda path: int(re.search(r"epoch=(\d+)", Path(path).name).group(1)),
    )


def configure(original, args, implementation, batch_size):
    cfg = copy.deepcopy(original)
    cfg.config = args.config_name
    cfg.DATA.mode = "cv"
    cfg.DATA.fold = args.fold
    cfg.DATA.train_dataloader.shuffle = False
    cfg.DATA.train_dataloader.num_workers = 0
    cfg.DATA.train_dataloader.batch_size = batch_size
    cfg.MODEL.compact_neighbor_tokens = True
    cfg.MODEL.num_genes = cfg.DATA.num_outputs
    cfg.MODEL.gene_path = (
        f"{cfg.DATA.data_dir}/{cfg.DATA.gene_type}_{cfg.DATA.num_genes}genes.json"
    )
    if implementation == "level2":
        cfg.DATA.neighbor_compact_dir = str(Path(args.compact_dir).resolve())
    else:
        cfg.DATA.pop("neighbor_compact_dir", None)
    return cfg


def move_batch(batch, device, dataset):
    moved = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    moved["pid"] = moved["pid"].view(-1)
    moved["sid"] = moved["sid"].view(-1)
    moved["dataset"] = dataset
    return moved


def measure(model, batch, warmup, iterations, device):
    model.train(True)

    def step():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = model(**batch, phase="train")["loss"]
        loss.backward()
        model.zero_grad(set_to_none=True)

    for _ in range(warmup):
        step()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        step()
    end.record()
    torch.cuda.synchronize(device)
    elapsed = start.elapsed_time(end)
    return {
        "milliseconds_per_step": elapsed / iterations,
        "spots_per_second": batch["img"].shape[0] * iterations / (elapsed / 1000),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
    }


def main():
    args = parse_args()
    device = torch.device("cuda")
    log_dir = Path("logs") / args.config_name / args.timestamp
    original = load_config(log_dir / "config.yaml")
    checkpoint = best_checkpoint(log_dir, args.fold)
    rows = []
    for implementation in ("level1", "level2"):
        for requested_batch_size in args.batch_sizes:
            cfg = configure(original, args, implementation, requested_batch_size)
            fix_seed(cfg.GENERAL.seed)
            dm = DataInterface(dataset_name=cfg.DATA.dataset_name, data_config=cfg.DATA)
            dm.setup("fit")
            cpu_batch = next(iter(dm.train_dataloader()))
            batch = move_batch(cpu_batch, device, dm.train_dataset)
            interface = ModelInterface.load_from_checkpoint(
                checkpoint,
                model_name=cfg.MODEL.model_name,
                config=cfg,
            ).to(device)
            model = interface.model
            for repeat in range(1, args.repeats + 1):
                try:
                    result = measure(
                        model, batch, args.warmup, args.iterations, device
                    )
                    status = "ok"
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    result = {
                        "milliseconds_per_step": "",
                        "spots_per_second": "",
                        "peak_allocated_mib": "",
                    }
                    status = "oom"
                row = {
                    "implementation": implementation,
                    "requested_batch_size": requested_batch_size,
                    "actual_batch_size": cpu_batch["img"].shape[0],
                    "neighbor_tokens": cpu_batch["neighbor_emb"].shape[1],
                    "repeat": repeat,
                    "status": status,
                    **result,
                }
                rows.append(row)
                print(row, flush=True)
                if status == "oom":
                    break
            del interface, model, batch, dm
            torch.cuda.empty_cache()
            if rows[-1]["status"] == "oom":
                break

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
