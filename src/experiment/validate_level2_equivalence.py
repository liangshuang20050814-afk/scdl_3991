"""Compare Level 1 and Level 2 logits using the same trained checkpoint."""

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
    parser.add_argument("--batch-size", type=int, default=128)
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


def configure(cfg, args, compact_dir=None):
    cfg = copy.deepcopy(cfg)
    cfg.config = args.config_name
    cfg.DATA.mode = "cv"
    cfg.DATA.fold = args.fold
    cfg.DATA.train_dataloader.shuffle = False
    cfg.DATA.train_dataloader.num_workers = 0
    cfg.DATA.train_dataloader.batch_size = args.batch_size
    cfg.MODEL.compact_neighbor_tokens = True
    cfg.MODEL.num_genes = cfg.DATA.num_outputs
    cfg.MODEL.gene_path = (
        f"{cfg.DATA.data_dir}/{cfg.DATA.gene_type}_{cfg.DATA.num_genes}genes.json"
    )
    if compact_dir is not None:
        cfg.DATA.neighbor_compact_dir = str(Path(compact_dir).resolve())
    else:
        cfg.DATA.pop("neighbor_compact_dir", None)
    return cfg


def first_batch(cfg, seed):
    fix_seed(seed)
    dm = DataInterface(dataset_name=cfg.DATA.dataset_name, data_config=cfg.DATA)
    dm.setup("fit")
    return dm, next(iter(dm.train_dataloader()))


def move_batch(batch, device, dataset):
    moved = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    moved["pid"] = moved["pid"].view(-1)
    moved["sid"] = moved["sid"].view(-1)
    moved["dataset"] = dataset
    return moved


def main():
    args = parse_args()
    device = torch.device("cuda")
    log_dir = Path("logs") / args.config_name / args.timestamp
    original_cfg = load_config(log_dir / "config.yaml")
    level1_cfg = configure(original_cfg, args)
    level2_cfg = configure(original_cfg, args, args.compact_dir)
    checkpoint = best_checkpoint(log_dir, args.fold)

    level1_dm, level1_batch = first_batch(level1_cfg, original_cfg.GENERAL.seed)
    level2_dm, level2_batch = first_batch(level2_cfg, original_cfg.GENERAL.seed)
    if not torch.equal(level1_batch["img"], level2_batch["img"]):
        raise ValueError("Target images differ between Level 1 and Level 2")
    if not torch.equal(level1_batch["label"], level2_batch["label"]):
        raise ValueError("Labels differ between Level 1 and Level 2")

    interface = ModelInterface.load_from_checkpoint(
        checkpoint,
        model_name=level1_cfg.MODEL.model_name,
        config=level1_cfg,
    ).to(device).eval()
    model = interface.model
    level1_batch = move_batch(level1_batch, device, level1_dm.train_dataset)
    level2_batch = move_batch(level2_batch, device, level2_dm.train_dataset)
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            torch.cuda.manual_seed_all(original_cfg.GENERAL.seed)
            level1_logits = model(**level1_batch, phase="train")["logits"]
            torch.cuda.manual_seed_all(original_cfg.GENERAL.seed)
            level2_logits = model(**level2_batch, phase="train")["logits"]
    difference = (level1_logits - level2_logits).abs()
    row = {
        "config": args.config_name,
        "timestamp": args.timestamp,
        "fold": args.fold,
        "batch_size": args.batch_size,
        "level1_tokens": level1_batch["neighbor_emb"].shape[1],
        "level2_tokens": level2_batch["neighbor_emb"].shape[1],
        "max_abs_logit_difference": float(difference.max().cpu()),
        "mean_abs_logit_difference": float(difference.mean().cpu()),
        "allclose_atol_1e-5": bool(torch.allclose(
            level1_logits, level2_logits, atol=1e-5, rtol=1e-5
        )),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    print(row)


if __name__ == "__main__":
    main()
