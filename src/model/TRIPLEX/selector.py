import torch
import torch.nn as nn
import torch.nn.functional as F


class _StraightThroughGate(torch.autograd.Function):
    """Use an exact hard gate forward and a soft-surrogate gradient backward."""

    @staticmethod
    def forward(ctx, hard_gate, soft_gate):
        return hard_gate

    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output


class TargetConditionedSelector(nn.Module):
    """Score context patches relative to the center target patch.

    The selector supports three experiment modes that share the same scorer:

    - ``soft``: all valid patches remain active with continuous gates.
    - ``straight_through``: forward uses a hard Top-K gate while backward uses
      the corresponding soft gate as a surrogate gradient.
    - ``frozen_hard``: deterministic hard Top-K selection for a scorer loaded
      from a separately trained soft-attention checkpoint.
    """

    VALID_MODES = {'soft', 'straight_through', 'frozen_hard'}

    def __init__(
            self,
            emb_dim,
            selector_dim=64,
            num_selected_patches=20,
            temperature=1.0,
            use_similarity_prior=False,
            similarity_prior_weight=1.0,
            learned_score_weight=1.0):
        super().__init__()

        if selector_dim < 1:
            raise ValueError("selector_dim must be at least 1")
        if num_selected_patches < 1:
            raise ValueError("num_selected_patches must be at least 1")
        if temperature <= 0:
            raise ValueError("selector temperature must be greater than 0")

        self.num_selected_patches = num_selected_patches
        self.temperature = temperature
        self.use_similarity_prior = use_similarity_prior
        self.similarity_prior_weight = similarity_prior_weight
        self.learned_score_weight = learned_score_weight

        self.norm = nn.LayerNorm(emb_dim)
        self.to_query = nn.Linear(emb_dim, selector_dim, bias=False)
        self.to_key = nn.Linear(emb_dim, selector_dim, bias=False)
        self.scale = selector_dim ** -0.5

    def forward(self, neighbor_emb, valid_mask, mode):
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"selector mode must be one of {sorted(self.VALID_MODES)}, "
                f"but got {mode}"
            )
        if neighbor_emb.ndim != 3:
            raise ValueError("neighbor_emb must have shape [B, N, D]")
        if valid_mask.ndim != 2:
            raise ValueError("valid_mask must have shape [B, N]")
        if neighbor_emb.shape[:2] != valid_mask.shape:
            raise ValueError(
                "neighbor_emb and valid_mask must have matching batch and "
                "token dimensions"
            )

        valid = valid_mask.bool()
        num_tokens = neighbor_emb.shape[1]
        center_idx = num_tokens // 2

        if not torch.all(valid[:, center_idx]):
            raise ValueError("The center target patch must be valid")

        normalized = self.norm(neighbor_emb)
        query = self.to_query(normalized[:, center_idx])
        keys = self.to_key(normalized)
        learned_scores = torch.einsum('bd,bnd->bn', query, keys) * self.scale

        scores = learned_scores * self.learned_score_weight
        if self.use_similarity_prior:
            unit_emb = F.normalize(neighbor_emb.float(), p=2, dim=-1)
            similarity = torch.einsum(
                'bd,bnd->bn',
                unit_emb[:, center_idx],
                unit_emb,
            ).to(scores.dtype)
            scores = scores + similarity * self.similarity_prior_weight

        context_valid = valid.clone()
        context_valid[:, center_idx] = False

        hard_gate = self._hard_topk_gate(scores, context_valid, center_idx)
        soft_gate, budget_loss = self._soft_budget_gate(
            scores,
            context_valid,
            center_idx,
        )

        if mode == 'soft':
            gate = soft_gate
        elif mode == 'straight_through' and self.training:
            gate = _StraightThroughGate.apply(hard_gate, soft_gate)
        else:
            gate = hard_gate

        gate = gate * valid.to(gate.dtype)
        selected_mask = hard_gate.bool() & valid

        return {
            'gate': gate,
            'hard_gate': hard_gate,
            'soft_gate': soft_gate,
            'selected_mask': selected_mask,
            'scores': scores,
            'budget_loss': budget_loss,
        }

    def _hard_topk_gate(self, scores, context_valid, center_idx):
        gate = torch.zeros_like(scores)
        gate[:, center_idx] = 1.0
        context_budget = self.num_selected_patches - 1

        for row in range(scores.shape[0]):
            candidates = torch.nonzero(
                context_valid[row],
                as_tuple=False,
            ).flatten()
            num_to_select = min(context_budget, candidates.numel())
            if num_to_select == 0:
                continue
            if num_to_select == candidates.numel():
                chosen = candidates
            else:
                candidate_scores = scores[row, candidates]
                order = torch.argsort(
                    candidate_scores,
                    descending=True,
                    stable=True,
                )
                chosen = candidates[order[:num_to_select]]
            gate[row, chosen] = 1.0

        return gate

    def _soft_budget_gate(self, scores, context_valid, center_idx):
        gate = torch.zeros_like(scores)
        gate[:, center_idx] = 1.0
        context_budget = self.num_selected_patches - 1
        budget_losses = []

        for row in range(scores.shape[0]):
            candidates = torch.nonzero(
                context_valid[row],
                as_tuple=False,
            ).flatten()
            num_to_select = min(context_budget, candidates.numel())
            if num_to_select == 0:
                continue
            if num_to_select == candidates.numel():
                gate[row, candidates] = 1.0
                continue

            candidate_scores = scores[row, candidates]
            top_scores = torch.topk(
                candidate_scores,
                k=num_to_select,
                sorted=True,
            ).values
            threshold = top_scores[-1].detach()
            soft_values = torch.sigmoid(
                (candidate_scores - threshold) / self.temperature
            )
            gate[row, candidates] = soft_values
            budget_losses.append(
                (soft_values.sum() - float(num_to_select)).pow(2)
            )

        if budget_losses:
            budget_loss = torch.stack(budget_losses).mean()
        else:
            budget_loss = scores.sum() * 0.0

        return gate, budget_loss


def load_selector_state(selector, checkpoint_path):
    """Load only selector parameters from a Lightning or plain checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)

    prefixes = (
        'model.neighbor_selector.',
        'neighbor_selector.',
    )
    selector_state = {}
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                selector_state[key[len(prefix):]] = value
                break

    if not selector_state:
        expected = ', '.join(prefixes)
        raise ValueError(
            f"No selector weights found in {checkpoint_path}. "
            f"Expected keys beginning with: {expected}"
        )

    selector.load_state_dict(selector_state, strict=True)


def freeze_selector(selector):
    selector.eval()
    for parameter in selector.parameters():
        parameter.requires_grad = False
