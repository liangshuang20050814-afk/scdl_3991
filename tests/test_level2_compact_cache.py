import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset.tri_dataset import TriDataset
from preprocess.build_compact_neighbor_cache import (
    make_selector,
    materialize_slide,
    validate_slide,
)


class Level2CompactCacheTest(unittest.TestCase):
    def test_materialized_cache_contains_only_selected_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.h5"
            destination = root / "compact" / "source.h5"
            embeddings = np.arange(3 * 25 * 4, dtype=np.float32).reshape(3, 25, 4)
            masks = np.ones((3, 25), dtype=np.uint8)
            masks[1, :22] = 0
            masks[1, 12] = 1
            with h5py.File(source, "w") as handle:
                handle.create_dataset("embeddings", data=embeddings)
                handle.create_dataset("mask_tb", data=masks)

            selector = make_selector("random", 5, 1)
            materialize_slide(source, destination, selector, 5, {})
            validate_slide(source, destination, selector, 5)

            with h5py.File(destination, "r") as handle:
                self.assertEqual(handle["embeddings"].shape, (3, 5, 4))
                self.assertEqual(handle["mask_tb"].shape, (3, 5))
                self.assertEqual(handle["token_indices"].shape, (3, 5))
                self.assertEqual(int(handle["mask_tb"][1].sum()), 4)

    def test_dataset_loads_compact_indices_without_reselection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compact_dir = root / "compact"
            compact_dir.mkdir()
            with h5py.File(compact_dir / "slide.h5", "w") as handle:
                handle.create_dataset("embeddings", data=np.ones((2, 5, 4), np.float32))
                handle.create_dataset("mask_tb", data=np.ones((2, 5), np.uint8))
                handle.create_dataset(
                    "token_indices",
                    data=np.array([[1, 4, 12, 19, 22]] * 2, np.int16),
                )

            dataset = object.__new__(TriDataset)
            dataset.emb_dir = str(root / "unused")
            dataset.model_name = "cigar"
            dataset.neighbor_compact_dir = str(compact_dir)
            embeddings, mask, indices = dataset.load_emb(
                "slide", emb_name="neighbor", idx=0
            )
            self.assertEqual(embeddings.shape, (5, 4))
            self.assertEqual(mask.shape, (5,))
            self.assertTrue(
                torch.equal(indices, torch.tensor([1, 4, 12, 19, 22]))
            )
