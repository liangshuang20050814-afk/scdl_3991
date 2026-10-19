import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from dataset.tri_dataset import TriDataset


class NeighborSelectionTest(unittest.TestCase):
    def make_dataset(self, method, selected=20, seed=2021):
        dataset = object.__new__(TriDataset)
        dataset.neighbor_selection = method
        dataset.num_selected_patches = selected
        dataset.selection_seed = seed
        return dataset

    def assert_valid_selection(self, selected, valid, expected_count):
        self.assertEqual(selected.dtype, valid.dtype)
        self.assertEqual(int(selected.sum()), expected_count)
        self.assertEqual(int(selected[12]), 1)
        self.assertFalse(torch.any(selected.bool() & ~valid.bool()))

    def test_scalar_index_types_are_supported(self):
        embeddings = torch.randn(25, 8)
        valid = torch.ones(25, dtype=torch.long)
        dataset = self.make_dataset('random20', seed=7)

        python_selection = dataset.select_neighbor_mask(
            embeddings, valid, 'slide', 3
        )
        numpy_selection = dataset.select_neighbor_mask(
            embeddings, valid, 'slide', np.int64(3)
        )
        zero_dimensional_selection = dataset.select_neighbor_mask(
            embeddings, valid, 'slide', np.array(3)
        )

        self.assertTrue(torch.equal(python_selection, numpy_selection))
        self.assertTrue(torch.equal(python_selection, zero_dimensional_selection))
        self.assert_valid_selection(python_selection, valid, 20)

    def test_batch_indices_and_all_static_methods(self):
        embeddings = torch.randn(2, 25, 8)
        valid = torch.ones(2, 25, dtype=torch.long)

        for method in ('random20', 'similarity20', 'diversity20'):
            with self.subTest(method=method):
                dataset = self.make_dataset(method)
                selected = dataset.select_neighbor_mask(
                    embeddings, valid, 'slide', torch.tensor([4, 5])
                )
                self.assertEqual(tuple(selected.shape), (2, 25))
                for row in range(2):
                    self.assert_valid_selection(selected[row], valid[row], 20)

    def test_edge_spot_keeps_every_valid_patch(self):
        embeddings = torch.randn(25, 8)
        valid = torch.zeros(25, dtype=torch.long)
        valid[[0, 6, 12, 18, 24]] = 1

        for method in ('random20', 'similarity20', 'diversity20'):
            with self.subTest(method=method):
                selected = self.make_dataset(method).select_neighbor_mask(
                    embeddings, valid, 'edge-slide', np.int32(2)
                )
                self.assertTrue(torch.equal(selected, valid))

    def test_random_selection_is_deterministic_per_spot(self):
        embeddings = torch.randn(25, 8)
        valid = torch.ones(25, dtype=torch.long)
        dataset = self.make_dataset('random20', seed=11)

        first = dataset.select_neighbor_mask(
            embeddings, valid, 'slide-a', np.int64(9)
        )
        second = dataset.select_neighbor_mask(
            embeddings, valid, 'slide-a', torch.tensor(9)
        )

        self.assertTrue(torch.equal(first, second))

    def test_invalid_center_patch_fails_early(self):
        embeddings = torch.randn(25, 8)
        valid = torch.ones(25, dtype=torch.long)
        valid[12] = 0

        with self.assertRaisesRegex(ValueError, "Center patch is invalid"):
            self.make_dataset('random20').select_neighbor_mask(
                embeddings, valid, 'slide', 0
            )

    def test_generic_method_names_support_multiple_budgets(self):
        embeddings = torch.randn(25, 8)
        valid = torch.ones(25, dtype=torch.long)

        for method in ('random', 'similarity', 'diversity'):
            for selected in (5, 10, 15):
                result = self.make_dataset(
                    method, seed=7, selected=selected
                ).select_neighbor_mask(embeddings, valid, 'slide', 3)
                self.assert_valid_selection(result, valid, selected)

    def test_embedding_loader_uses_configured_model_name(self):
        with tempfile.TemporaryDirectory() as directory:
            embedding_root = Path(directory) / 'neighbor' / 'cigar'
            embedding_root.mkdir(parents=True)
            embedding_path = embedding_root / 'sample.h5'

            with h5py.File(embedding_path, 'w') as handle:
                handle.create_dataset(
                    'features', data=np.zeros((2, 25, 4), dtype=np.float32)
                )
                handle.create_dataset(
                    'mask_tb', data=np.ones((2, 25), dtype=np.int64)
                )

            dataset = object.__new__(TriDataset)
            dataset.emb_dir = directory
            dataset.model_name = 'cigar'
            embeddings, valid = dataset.load_emb(
                'sample', emb_name='neighbor', idx=0
            )

            self.assertEqual(tuple(embeddings.shape), (25, 4))
            self.assertEqual(tuple(valid.shape), (25,))


if __name__ == '__main__':
    unittest.main()
