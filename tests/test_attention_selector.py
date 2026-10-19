import tempfile
import unittest
from pathlib import Path

import torch

from model.TRIPLEX.selector import (
    TargetConditionedSelector,
    freeze_selector,
    load_selector_state,
)


class AttentionSelectorTest(unittest.TestCase):
    def make_inputs(self, batch_size=2, num_tokens=9, emb_dim=8):
        generator = torch.Generator().manual_seed(17)
        embeddings = torch.randn(
            batch_size, num_tokens, emb_dim, generator=generator
        )
        valid = torch.ones(batch_size, num_tokens, dtype=torch.long)
        return embeddings, valid

    def make_selector(self, selected=5):
        torch.manual_seed(23)
        return TargetConditionedSelector(
            emb_dim=8,
            selector_dim=4,
            num_selected_patches=selected,
            temperature=1.0,
        )

    def test_hard_modes_keep_center_and_exact_budget(self):
        embeddings, valid = self.make_inputs()
        center = embeddings.shape[1] // 2

        for mode in ('straight_through', 'frozen_hard'):
            with self.subTest(mode=mode):
                selector = self.make_selector(selected=5)
                selector.eval()
                result = selector(embeddings, valid, mode)
                self.assertTrue(torch.all(result['gate'][:, center] == 1))
                self.assertTrue(torch.all(result['selected_mask'].sum(1) == 5))
                self.assertTrue(torch.equal(
                    result['gate'].bool(), result['selected_mask']
                ))

    def test_edge_spot_keeps_every_valid_patch(self):
        embeddings, valid = self.make_inputs(batch_size=1)
        valid.zero_()
        valid[0, [0, 2, 4, 7]] = 1

        for mode in ('soft', 'straight_through', 'frozen_hard'):
            with self.subTest(mode=mode):
                selector = self.make_selector(selected=5)
                selector.eval()
                result = selector(embeddings, valid, mode)
                self.assertTrue(torch.equal(result['gate'].bool(), valid.bool()))
                self.assertEqual(float(result['budget_loss']), 0.0)

    def test_soft_gate_is_continuous_and_respects_valid_mask(self):
        embeddings, valid = self.make_inputs(batch_size=1)
        valid[0, 0] = 0
        selector = self.make_selector(selected=5)
        result = selector(embeddings, valid, 'soft')
        center = embeddings.shape[1] // 2

        self.assertEqual(float(result['gate'][0, center]), 1.0)
        self.assertEqual(float(result['gate'][0, 0]), 0.0)
        context = result['gate'][0, valid[0].bool()]
        self.assertTrue(torch.any((context > 0) & (context < 1)))
        self.assertTrue(torch.isfinite(result['budget_loss']))

    def test_straight_through_gate_propagates_selector_gradients(self):
        embeddings, valid = self.make_inputs(batch_size=2)
        selector = self.make_selector(selected=5)
        selector.train()
        result = selector(embeddings, valid, 'straight_through')
        weights = torch.arange(
            result['gate'].numel(), dtype=result['gate'].dtype
        ).reshape_as(result['gate'])
        (result['gate'] * weights).sum().backward()

        gradients = [
            parameter.grad for parameter in selector.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(any(torch.any(gradient != 0) for gradient in gradients))

    def test_selector_checkpoint_load_and_freeze(self):
        source = self.make_selector(selected=5)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / 'selector.ckpt'
            state = {
                f'model.neighbor_selector.{key}': value.clone()
                for key, value in source.state_dict().items()
            }
            torch.save({'state_dict': state}, checkpoint_path)

            target = self.make_selector(selected=5)
            for parameter in target.parameters():
                torch.nn.init.zeros_(parameter)
            load_selector_state(target, checkpoint_path)

            for key, value in source.state_dict().items():
                self.assertTrue(torch.equal(value, target.state_dict()[key]))

            freeze_selector(target)
            self.assertFalse(target.training)
            self.assertTrue(all(
                not parameter.requires_grad for parameter in target.parameters()
            ))

    def test_invalid_center_patch_fails(self):
        embeddings, valid = self.make_inputs(batch_size=1)
        valid[0, embeddings.shape[1] // 2] = 0
        with self.assertRaisesRegex(ValueError, 'center target patch'):
            self.make_selector()(embeddings, valid, 'soft')


if __name__ == '__main__':
    unittest.main()
