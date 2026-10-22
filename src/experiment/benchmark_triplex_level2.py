"""Benchmark mask, Level 1, and physically compact Level 2 inputs."""

import argparse
import copy
import csv
from glob import glob
from pathlib import Path
import re
import time

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
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--loader-batches", type=int, default=30)
    parser.add_argument("--pipeline-batches", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def best_checkpoint(log_dir, fold):
    paths = [
        path for path in glob(str(log_dir / f"fold{fold}" / "*.ckpt"))
        if Path(path).name != "last.ckpt"
    ]
    if not paths:
        raise FileNotFoundError(f"No checkpoint in {log_dir}/fold{fold}")

    def epoch(path):
        match = re.search(r"epoch=(\d+)", Path(path).name)
        return int(match.group(1)) if match else -1

    return max(paths, key=epoch)


def configure(original, args, implementation):
    cfg = copy.deepcopy(original)
    cfg.config = args.config_name
    cfg.DATA.mode = "cv"
    cfg.DATA.fold = args.fold
    cfg.DATA.train_dataloader.shuffle = False
    cfg.MODEL.num_genes = cfg.DATA.num_outputs
    cfg.MODEL.gene_path = (
        f"{cfg.DATA.data_dir}/{cfg.DATA.gene_type}_{cfg.DATA.num_genes}genes.json"
    )
    cfg.MODEL.compact_neighbor_tokens = implementation != "mask"
    if implementation == "level2":
        cfg.DATA.neighbor_compact_dir = str(Path(args.compact_dir).resolve())
    else:
        cfg.DATA.pop("neighbor_compact_dir", None)
    return cfg


def move_batch(batch, device, dataset):
    moved = {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    moved["pid"] = moved["pid"].view(-1)
    moved["sid"] = moved["sid"].view(-1)
    moved["dataset"] = dataset
    return moved


def measure_loader(dm, batches):
    loader = dm.train_dataloader()
    iterator = iter(loader)
    spots = 0
    start = time.perf_counter()
    for _ in range(batches):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        spots += batch["img"].shape[0]
    elapsed = time.perf_counter() - start
    return elapsed, spots


def measure_transfer(cpu_batch, device, warmup, iterations):
    tensors = [value for value in cpu_batch.values() if torch.is_tensor(value)]
    destinations = [torch.empty_like(value, device=device) for value in tensors]
    for _ in range(warmup):
        for destination, source in zip(destinations, tensors):
            destination.copy_(source, non_blocking=False)
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        for destination, source in zip(destinations, tensors):
            destination.copy_(source, non_blocking=False)
    end.record()
    torch.cuda.synchronize(device)
    elapsed_ms = start.elapsed_time(end)
    batch_size = cpu_batch["img"].shape[0]
    return {
        "milliseconds_per_step": elapsed_ms / iterations,
        "spots_per_second": batch_size * iterations / (elapsed_ms / 1000),
        "peak_allocated_mib": "",
        "incremental_peak_mib": "",
    }


def measure_pipeline(model, dm, batches, device):
    model.train(True)
    iterator = iter(dm.train_dataloader())
    spots = 0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    completed = 0
    for _ in range(batches):
        try:
            cpu_batch = next(iterator)
        except StopIteration:
            break
        gpu_batch = move_batch(cpu_batch, device, dm.train_dataset)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = model(**gpu_batch, phase="train")["loss"]
        loss.backward()
        model.zero_grad(set_to_none=True)
        spots += cpu_batch["img"].shape[0]
        completed += 1
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated(device)
    return {
        "milliseconds_per_step": elapsed * 1000 / completed,
        "spots_per_second": spots / elapsed,
        "peak_allocated_mib": peak / (1024 ** 2),
        "incremental_peak_mib": "",
    }


def measure_model(model, batch, training, warmup, iterations, device):
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
    peak = torch.cuda.max_memory_allocated(device)
    return {
        "milliseconds_per_step": elapsed_ms / iterations,
        "spots_per_second": batch["img"].shape[0] * iterations / (elapsed_ms / 1000),
        "peak_allocated_mib": peak / (1024 ** 2),
        "incremental_peak_mib": (peak - memory_before) / (1024 ** 2),
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    log_dir = Path("logs") / args.config_name / args.timestamp
    original_cfg = load_config(log_dir / "config.yaml")
    checkpoint = best_checkpoint(log_dir, args.fold)
    rows = []

    for implementation in ("mask", "level1", "level2"):
        cfg = configure(original_cfg, args, implementation)
        fix_seed(cfg.GENERAL.seed)
        dm = DataInterface(dataset_name=cfg.DATA.dataset_name, data_config=cfg.DATA)
        dm.setup("fit")
        first_batch = next(iter(dm.train_dataloader()))
        neighbor_payload = (
            first_batch["neighbor_emb"].numel()
            * first_batch["neighbor_emb"].element_size()
            / (1024 ** 2)
        )
        total_payload = sum(
            value.numel() * value.element_size()
            for value in first_batch.values()
            if torch.is_tensor(value)
        ) / (1024 ** 2)
        gpu_batch = move_batch(first_batch, device, dm.train_dataset)
        interface = ModelInterface.load_from_checkpoint(
            checkpoint,
            model_name=cfg.MODEL.model_name,
            config=cfg,
        ).to(device)
        model = interface.model

        for repeat in range(1, args.repeats + 1):
            elapsed, spots = measure_loader(dm, args.loader_batches)
            rows.append({
                "config": args.config_name,
                "implementation": implementation,
                "section": "dataloader",
                "phase": "load",
                "repeat": repeat,
                "batch_size": first_batch["img"].shape[0],
                "neighbor_tokens": first_batch["neighbor_emb"].shape[1],
                "neighbor_payload_mib": neighbor_payload,
                "total_payload_mib": total_payload,
                "milliseconds_per_step": elapsed * 1000 / args.loader_batches,
                "spots_per_second": spots / elapsed,
                "peak_allocated_mib": "",
                "incremental_peak_mib": "",
            })
            transfer_result = measure_transfer(
                first_batch, device, args.warmup, args.iterations
            )
            rows.append({
                "config": args.config_name,
                "implementation": implementation,
                "section": "transfer",
                "phase": "cpu_to_gpu",
                "repeat": repeat,
                "batch_size": first_batch["img"].shape[0],
                "neighbor_tokens": first_batch["neighbor_emb"].shape[1],
                "neighbor_payload_mib": neighbor_payload,
                "total_payload_mib": total_payload,
                **transfer_result,
            })
            for training in (False, True):
                result = measure_model(
                    model,
                    gpu_batch,
                    training,
                    args.warmup,
                    args.iterations,
                    device,
                )
                rows.append({
                    "config": args.config_name,
                    "implementation": implementation,
                    "section": "model",
                    "phase": "training" if training else "forward",
                    "repeat": repeat,
                    "batch_size": first_batch["img"].shape[0],
                    "neighbor_tokens": first_batch["neighbor_emb"].shape[1],
                    "neighbor_payload_mib": neighbor_payload,
                    "total_payload_mib": total_payload,
                    **result,
                })
                print(rows[-1], flush=True)
            pipeline_result = measure_pipeline(
                model, dm, args.pipeline_batches, device
            )
            rows.append({
                "config": args.config_name,
                "implementation": implementation,
                "section": "pipeline",
                "phase": "load_transfer_train",
                "repeat": repeat,
                "batch_size": first_batch["img"].shape[0],
                "neighbor_tokens": first_batch["neighbor_emb"].shape[1],
                "neighbor_payload_mib": neighbor_payload,
                "total_payload_mib": total_payload,
                **pipeline_result,
            })
        del interface, model, gpu_batch, dm
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
