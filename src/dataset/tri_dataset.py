
import hashlib
import os

import numpy as np
from scipy import sparse
import h5py
import torch
import torch.nn.functional as F

from dataset.base_dataset import STDataset


class TriDataset(STDataset):
    STATIC_SELECTION_METHODS = {
        'random',
        'random20',
        'similarity',
        'similarity20',
        'diversity',
        'diversity20',
    }
    MODEL_SELECTION_METHODS = {
        'attention_soft',
        'attention_st',
        'attention_st20',
        'attention_twostage',
        'attention_twostage20',
    }

    def __init__(self, 
                mode: str,
                phase: str,
                fold: int,
                data_dir: str,
                gene_type: str = 'mean',
                num_genes: int = 1000,
                num_outputs: int = 300,
                normalize: bool = True,
                cpm: bool = False,
                smooth: bool = False,
                data_id: str = None,
                model_name: str = 'uni_v1',
                neighbor_selection: str = 'all',
                num_selected_patches: int = 20,
                selection_seed: int = 2021,
                neighbor_compact_dir: str = None,
                ):
        super(TriDataset, self).__init__(
                                mode=mode,
                                phase=phase,
                                fold=fold,
                                data_dir=data_dir,
                                gene_type=gene_type,
                                num_genes=num_genes,
                                num_outputs=num_outputs,
                                normalize=normalize,
                                cpm=cpm,
                                smooth=smooth,
                                data_id=data_id)
    
        valid_selection_methods = [
            'all',
            'random',
            'random20',
            'similarity',
            'similarity20',
            'diversity',
            'diversity20',
            'attention_soft',
            'attention_st',
            'attention_st20',
            'attention_twostage',
            'attention_twostage20',
        ]
        if neighbor_selection not in valid_selection_methods:
            raise ValueError(
                f"neighbor_selection must be one of {valid_selection_methods}, "
                f"but got {neighbor_selection}"
            )
        if num_selected_patches < 1:
            raise ValueError("num_selected_patches must be at least 1")

        self.emb_dir = f"{data_dir}/emb"
        self.model_name = model_name
        self.neighbor_selection = neighbor_selection
        self.num_selected_patches = num_selected_patches
        self.selection_seed = selection_seed
        self.neighbor_compact_dir = neighbor_compact_dir
        
        if phase == 'train':
            
            self.global_data = {_id: self.load_emb(_id, emb_name='global', model_name=model_name, return_crds=True) \
                for _id in self.ids}
            self.pos_dict = {_id: data[1] \
                for _id, data in self.global_data.items()}
            self.global_embs = {_id: data[0] \
                for _id, data in self.global_data.items()}
        
        if mode == 'inference':
            self.global_emb, self.position = self.load_emb(self.name, emb_name='global', return_crds=True)
            
        
    def __getitem__(self, index):
        data = {}
        
        if self.phase == 'train':
            i = 0
            while index >= self.cumlen[i]:
                i += 1
            idx = index
            if i > 0:
                idx = index - self.cumlen[i-1]

            name = self.int2id[i]
            img = self.load_img(name, idx)
            img = self.transforms(img)
            
            neighbor_result = self.load_emb(name, emb_name='neighbor', idx=idx)
            neighbor_emb, mask = neighbor_result[:2]
            token_indices = neighbor_result[2] if len(neighbor_result) == 3 else None
            if token_indices is None:
                mask = self.select_neighbor_mask(neighbor_emb, mask, name, idx)
            adata = self.adata_dict[name]
            expression = adata[idx].X
            expression = expression.toarray().squeeze(0) \
                if sparse.issparse(expression) else expression.squeeze(0)
            
                
            data['img'] = img
            data['mask'] = mask
            if self.neighbor_selection in self.STATIC_SELECTION_METHODS:
                data['neighbor_pool_mask'] = mask.clone()
            data['neighbor_emb'] = neighbor_emb
            if token_indices is not None:
                data['neighbor_token_indices'] = token_indices
            data['label'] = torch.FloatTensor(expression)
            data['pid'] = torch.LongTensor([i])
            data['sid'] = torch.LongTensor([idx])
            
        elif self.phase == 'test':
            if self.mode == 'inference':
                img = self.img[index]
                img = self.transforms(img)
                neighbor_result = self.load_emb(
                    self.name, emb_name='neighbor', idx=index
                )
                neighbor_emb, mask = neighbor_result[:2]
                token_indices = (
                    neighbor_result[2] if len(neighbor_result) == 3 else None
                )
                if token_indices is None:
                    mask = self.select_neighbor_mask(
                        neighbor_emb,
                        mask,
                        self.name,
                        index,
                    )
                # global_emb = self.load_emb(self.name, emb_name='global')
                # pos = np.load(f"{self.data_dir}/pos/{self.name}.npy")
                data['sid'] = torch.LongTensor([index])
            else:
                name = self.int2id[index]
                img = self.load_img(name)
                img = torch.stack([self.transforms(im) for im in img], dim=0)
                
                neighbor_result = self.load_emb(name, emb_name='neighbor')
                neighbor_emb, mask = neighbor_result[:2]
                token_indices = (
                    neighbor_result[2] if len(neighbor_result) == 3 else None
                )
                if token_indices is None:
                    mask = self.select_neighbor_mask(
                        neighbor_emb,
                        mask,
                        name,
                        torch.arange(mask.shape[0]),
                    )
                global_emb, pos = self.load_emb(name, emb_name='global', return_crds=True)
            
                if os.path.isfile(f"{self.st_dir}/{name}.h5ad"):
                    adata = self.load_st(name, self.genes, **self.norm_param)
                    
                    expression = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
                    data['label'] = torch.FloatTensor(expression) 
                    
                data['position'] = pos
                data['global_emb'] = global_emb
                    
            
            data['img'] = img
            data['mask'] = mask
            if self.neighbor_selection in self.STATIC_SELECTION_METHODS:
                data['neighbor_pool_mask'] = mask.clone()
            data['neighbor_emb'] = neighbor_emb
            if token_indices is not None:
                data['neighbor_token_indices'] = token_indices
            
        return data

    def select_neighbor_mask(
            self,
            neighbor_emb,
            mask,
            slide_id,
            spot_indices):
        """Select a subset of valid neighbor patches.

        All selection methods retain the center target patch and choose at most
        ``num_selected_patches - 1`` valid context patches. Random-K uses a
        stable per-spot seed, Similarity-K ranks candidates by cosine
        similarity to the center, and Diversity-K applies center-initialized
        k-center greedy selection with cosine distance.
        """
        if (
            self.neighbor_selection == 'all'
            or self.neighbor_selection in self.MODEL_SELECTION_METHODS
        ):
            return mask

        squeeze_output = mask.ndim == 1
        masks = mask.unsqueeze(0) if squeeze_output else mask
        embeddings = (
            neighbor_emb.unsqueeze(0)
            if neighbor_emb.ndim == 2
            else neighbor_emb
        )

        if masks.ndim != 2:
            raise ValueError("mask must have shape [N] or [B, N]")
        if embeddings.ndim != 3:
            raise ValueError(
                "neighbor_emb must have shape [N, D] or [B, N, D]"
            )
        if embeddings.shape[:2] != masks.shape:
            raise ValueError(
                "neighbor_emb and mask must have matching batch and token "
                f"dimensions, got {tuple(embeddings.shape)} and "
                f"{tuple(masks.shape)}"
            )

        indices = self._normalize_spot_indices(spot_indices)

        if len(indices) != masks.shape[0]:
            raise ValueError(
                "spot_indices must contain one index for each neighbor mask"
            )

        selected_masks = torch.zeros_like(masks)
        num_tokens = masks.shape[-1]
        center_idx = num_tokens // 2
        num_context_patches = min(
            self.num_selected_patches - 1,
            num_tokens - 1,
        )

        for row, spot_idx in enumerate(indices):
            valid = masks[row].bool()

            # The center is the target patch and is never removed.
            if not valid[center_idx]:
                raise ValueError(
                    f"Center patch is invalid for slide {slide_id}, spot {spot_idx}"
                )
            selected_masks[row, center_idx] = 1

            candidate_mask = valid.clone()
            candidate_mask[center_idx] = False
            candidates = torch.nonzero(candidate_mask, as_tuple=False).flatten()
            num_to_select = min(num_context_patches, candidates.numel())

            if num_to_select == 0:
                continue

            if num_to_select == candidates.numel():
                chosen = candidates
            elif self.neighbor_selection in {'random', 'random20'}:
                chosen = self._select_random_candidates(
                    candidates,
                    num_to_select,
                    slide_id,
                    spot_idx,
                )
            elif self.neighbor_selection in {'similarity', 'similarity20'}:
                chosen = self._select_similar_candidates(
                    embeddings[row],
                    candidates,
                    center_idx,
                    num_to_select,
                )
            elif self.neighbor_selection in {'diversity', 'diversity20'}:
                chosen = self._select_diverse_candidates(
                    embeddings[row],
                    candidates,
                    center_idx,
                    num_to_select,
                )
            else:
                raise RuntimeError(
                    f"Unsupported neighbor selection method: "
                    f"{self.neighbor_selection}"
                )

            selected_masks[row, chosen] = 1

        return selected_masks.squeeze(0) if squeeze_output else selected_masks

    @staticmethod
    def _normalize_spot_indices(spot_indices):
        """Return spot indices as Python integers for stable selection seeds."""
        if isinstance(spot_indices, (int, np.integer)):
            indices = [spot_indices]
        elif torch.is_tensor(spot_indices):
            indices = spot_indices.detach().cpu().reshape(-1).tolist()
        elif isinstance(spot_indices, np.ndarray):
            indices = spot_indices.reshape(-1).tolist()
        else:
            try:
                indices = list(spot_indices)
            except TypeError as error:
                raise TypeError(
                    "spot_indices must be an integer, tensor, NumPy array, "
                    "or iterable of integers"
                ) from error

        try:
            return [int(index) for index in indices]
        except (TypeError, ValueError) as error:
            raise TypeError("spot_indices must contain only integers") from error

    def _select_random_candidates(
            self,
            candidates,
            num_to_select,
            slide_id,
            spot_idx):
        seed_material = f"{self.selection_seed}:{slide_id}:{int(spot_idx)}"
        digest = hashlib.sha256(seed_material.encode('utf-8')).digest()
        spot_seed = int.from_bytes(digest[:8], byteorder='little')
        spot_seed %= (2**63 - 1)

        generator = torch.Generator(device='cpu')
        generator.manual_seed(spot_seed)
        order = torch.randperm(candidates.numel(), generator=generator)
        return candidates[order[:num_to_select]]

    @staticmethod
    def _select_similar_candidates(
            embeddings,
            candidates,
            center_idx,
            num_to_select):
        """Return candidates most cosine-similar to the center patch."""
        normalized = F.normalize(embeddings.float(), p=2, dim=-1)
        similarities = normalized[candidates] @ normalized[center_idx]

        # Candidates are in ascending token-index order. Stable sorting makes
        # token index the deterministic tie-breaker for equal similarities.
        order = torch.argsort(similarities, descending=True, stable=True)
        return candidates[order[:num_to_select]]

    @staticmethod
    def _select_diverse_candidates(
            embeddings,
            candidates,
            center_idx,
            num_to_select):
        """Apply center-initialized k-center greedy with cosine distance."""
        normalized = F.normalize(embeddings.float(), p=2, dim=-1)
        remaining = candidates.clone()

        similarities_to_center = normalized[remaining] @ normalized[center_idx]
        min_distances = 1.0 - similarities_to_center.clamp(-1.0, 1.0)
        chosen = []

        for _ in range(num_to_select):
            # torch.argmax returns the first maximum. Because ``remaining`` is
            # ordered by token index, equal distances have a stable tie-break.
            best_position = torch.argmax(min_distances)
            chosen_idx = remaining[best_position]
            chosen.append(chosen_idx)

            keep = torch.ones(
                remaining.numel(),
                dtype=torch.bool,
                device=remaining.device,
            )
            keep[best_position] = False
            remaining = remaining[keep]
            min_distances = min_distances[keep]

            if remaining.numel() == 0:
                break

            similarities_to_chosen = normalized[remaining] @ normalized[chosen_idx]
            distances_to_chosen = 1.0 - similarities_to_chosen.clamp(-1.0, 1.0)
            min_distances = torch.minimum(
                min_distances,
                distances_to_chosen,
            )

        return torch.stack(chosen)
        
    def load_emb(self, name: str, emb_name: str = 'global', idx: int = None, model_name: str = None, return_crds=False):
        if emb_name not in ['global', 'neighbor']:
            raise ValueError(f"emb_name must be 'global' or 'neighbor', but got {emb_name}")

        if model_name is None:
            model_name = self.model_name
        
        compact_dir = getattr(self, 'neighbor_compact_dir', None)
        if emb_name == 'neighbor' and compact_dir:
            path = f"{compact_dir}/{name}.h5"
        else:
            path = f"{self.emb_dir}/{emb_name}/{model_name}/{name}.h5"
        
        with h5py.File(path, 'r') as f:
            if 'embeddings'in f:
                emb = f['embeddings'][idx] if idx is not None else f['embeddings'][:]
            else:
                emb = f['features'][idx] if idx is not None else f['features'][:]
                
            emb = torch.Tensor(emb)
            
            if emb_name == 'neighbor':
                mask = f['mask_tb'][idx] if idx is not None else f['mask_tb'][:]
                mask = torch.LongTensor(mask)
                if 'token_indices' in f:
                    token_indices = (
                        f['token_indices'][idx]
                        if idx is not None
                        else f['token_indices'][:]
                    )
                    token_indices = torch.LongTensor(token_indices)
                    return emb, mask, token_indices
                # return emb, mask
                if return_crds:
                    coords = f['coords'][idx] if idx is not None else f['coords'][:]
                    coords = torch.Tensor(coords)
                    return emb, mask, coords
                else:
                    return emb, mask
            else:
                if return_crds:
                    coords = f['coords'][idx] if idx is not None else f['coords'][:]
                    coords = torch.Tensor(coords)
                    return emb, coords
                else:
                    return emb
                
