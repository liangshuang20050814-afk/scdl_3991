import unittest

import torch

from model.TRIPLEX.TRIPLEX import TRIPLEX
from model.TRIPLEX.module import NeighborEncoder
from model.TRIPLEX.selector import TargetConditionedSelector


class CompactNeighborTokensTest(unittest.TestCase):
    def test_compaction_preserves_original_indices_and_padding(self):
        embeddings = torch.arange(2 * 9 * 3, dtype=torch.float32).reshape(2, 9, 3)
        selected = torch.zeros(2, 9, dtype=torch.long)
        selected[0, [1, 4, 8]] = 1
        selected[1, [4, 7]] = 1

        compact, mask, gate, pool_mask, indices = (
            TRIPLEX._compact_neighbor_tokens(embeddings, selected)
        )

        self.assertEqual(tuple(compact.shape), (2, 3, 3))
        self.assertTrue(torch.equal(indices[0], torch.tensor([1, 4, 8])))
        self.assertTrue(torch.equal(indices[1], torch.tensor([4, 7, 0])))
        self.assertTrue(torch.equal(mask, torch.tensor([[1, 1, 1], [1, 1, 0]])))
        self.assertIsNone(gate)
        self.assertTrue(torch.equal(pool_mask, mask.bool()))
        self.assertTrue(torch.equal(compact[0], embeddings[0, [1, 4, 8]]))

    def test_neighbor_encoder_matches_masked_forward_for_selected_tokens(self):
        torch.manual_seed(31)
        embeddings = torch.randn(2, 25, 16)
        selected = torch.zeros(2, 25, dtype=torch.long)
        selected[0, [0, 4, 12, 18, 24]] = 1
        selected[1, [2, 7, 12, 17, 22]] = 1

        encoder = NeighborEncoder(
            emb_dim=16,
            depth=2,
            heads=4,
            mlp_dim=32,
            dropout=0.0,
            resolution=(5, 5),
        )
        encoder.eval()

        with torch.no_grad():
            masked_output = encoder(embeddings.clone(), mask=selected)
            compact, compact_mask, _, _, indices = (
                TRIPLEX._compact_neighbor_tokens(embeddings, selected)
            )
            compact_output = encoder(
                compact,
                mask=compact_mask,
                token_indices=indices,
            )
            expected = masked_output.gather(
                1, indices.unsqueeze(-1).expand_as(compact_output)
            )

        self.assertTrue(torch.allclose(compact_output, expected, atol=1e-5, rtol=1e-5))

    def test_straight_through_selector_keeps_gradients_after_compaction(self):
        torch.manual_seed(37)
        embeddings = torch.randn(2, 9, 8)
        valid = torch.ones(2, 9, dtype=torch.long)
        selector = TargetConditionedSelector(
            emb_dim=8,
            selector_dim=4,
            num_selected_patches=5,
        )
        selector.train()
        result = selector(embeddings, valid, "straight_through")
        compact, _, compact_gate, _, _ = TRIPLEX._compact_neighbor_tokens(
            embeddings,
            result["selected_mask"],
            selector_gate=result["gate"],
            pool_mask=result["gate"],
        )
        (compact * compact_gate.unsqueeze(-1)).sum().backward()

        gradients = [
            parameter.grad
            for parameter in selector.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(any(torch.any(gradient != 0) for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
