"""Benchmark the TRIPLEX path affected by neighbor-token compaction.

This intentionally holds target and global tokens fixed and measures the
NeighborEncoder plus FusionEncoder. It isolates the compute changed by K from
unrelated image/global encoding work.
"""

import argparse
import csv
from pathlib import Path

import torch

from model.TRIPLEX.TRIPLEX import TRIPLEX
from model.TRIPLEX.module import FusionEncoder, NeighborEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def selected_mask(batch_size, budget, device):
    mask = torch.zeros(batch_size, 25, dtype=torch.long, device=device)
    center = 12
    candidates = [index for index in range(25) if index != center]
    chosen = sorted([center, *candidates[:budget - 1]])
    mask[:, chosen] = 1
    return mask


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(
        neighbor_encoder,
        fusion_encoder,
        embeddings,
        target_tokens,
        global_tokens,
        mask,
        compact,
        training,
        warmup,
        iterations,
        device):
    if compact:
        neighbor_input, active_mask, _, _, token_indices = (
            TRIPLEX._compact_neighbor_tokens(embeddings, mask)
        )
    else:
        neighbor_input = embeddings
        active_mask = mask
        token_indices = None

    neighbor_encoder.train(training)
    fusion_encoder.train(training)

    def step():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            neighbor_tokens = neighbor_encoder(
                neighbor_input,
                mask=active_mask,
                token_indices=token_indices,
            )
            output = fusion_encoder(
                target_tokens,
                neighbor_tokens,
                global_tokens,
                mask=active_mask,
            )
            loss = output.float().square().mean()
        if training:
            loss.backward()
            neighbor_encoder.zero_grad(set_to_none=True)
            fusion_encoder.zero_grad(set_to_none=True)

    for _ in range(warmup):
        if training:
            step()
        else:
            with torch.inference_mode():
                step()
    synchronize(device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        memory_before = torch.cuda.memory_allocated(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        memory_before = 0
        start = None
        end = None

    if device.type != "cuda":
        import time
        start_time = time.perf_counter()

    for _ in range(iterations):
        if training:
            step()
        else:
            with torch.inference_mode():
                step()

    if device.type == "cuda":
        end.record()
        synchronize(device)
        elapsed_ms = start.elapsed_time(end)
        peak_memory = torch.cuda.max_memory_allocated(device)
    else:
        import time
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        peak_memory = 0

    return {
        "milliseconds_per_step": elapsed_ms / iterations,
        "spots_per_second": embeddings.shape[0] * iterations / (elapsed_ms / 1000),
        "peak_allocated_mib": peak_memory / (1024 ** 2),
        "incremental_peak_mib": (peak_memory - memory_before) / (1024 ** 2),
        "input_tokens": neighbor_input.shape[1],
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    torch.manual_seed(2021)
    torch.set_float32_matmul_precision("high")
    emb_dim = 512
    neighbor_encoder = NeighborEncoder(
        emb_dim=emb_dim,
        depth=3,
        heads=16,
        mlp_dim=emb_dim,
        dropout=0.3,
        resolution=(5, 5),
    ).to(device)
    fusion_encoder = FusionEncoder(
        emb_dim=emb_dim,
        depth=1,
        heads=4,
        mlp_dim=emb_dim * 4,
        dropout=0.2,
    ).to(device)
    embeddings = torch.randn(args.batch_size, 25, emb_dim, device=device)
    target_tokens = torch.randn(args.batch_size, 49, emb_dim, device=device)
    global_tokens = torch.randn(args.batch_size, emb_dim, device=device)

    rows = []
    for training in (False, True):
        for budget in (5, 10, 15, 20, 25):
            mask = selected_mask(args.batch_size, budget, device)
            modes = (False,) if budget == 25 else (False, True)
            for compact in modes:
                for repeat in range(1, args.repeats + 1):
                    result = measure(
                        neighbor_encoder,
                        fusion_encoder,
                        embeddings,
                        target_tokens,
                        global_tokens,
                        mask,
                        compact,
                        training,
                        args.warmup,
                        args.iterations,
                        device,
                    )
                    rows.append({
                        "scope": "neighbor_plus_fusion",
                        "phase": "training" if training else "inference",
                        "implementation": "compact" if compact else "mask",
                        "budget": budget,
                        "batch_size": args.batch_size,
                        "repeat": repeat,
                        **result,
                    })
                    print(rows[-1])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
