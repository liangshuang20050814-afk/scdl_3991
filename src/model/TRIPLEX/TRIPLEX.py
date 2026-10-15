
import os 
import inspect
import importlib

import wget
import numpy as np
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl

import torch.nn.functional as F
from einops import rearrange

from model.TRIPLEX.module import ( GlobalEncoder, 
                                NeighborEncoder, 
                                FusionEncoder )
from model.TRIPLEX.selector import (
    TargetConditionedSelector,
    freeze_selector,
    load_selector_state,
)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def load_model_weights(ckpt: str):       
        """Load pretrained ResNet18 model without final fc layer.

        Args:
            path (str): path_for_pretrained_weight

        Returns:
            torchvision.models.resnet.ResNet: ResNet model with pretrained weight
        """
        
        resnet = torchvision.models.__dict__['resnet18'](weights=None)
        
        ckpt_dir = './weights/cigar'
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = f'{ckpt_dir}/{ckpt}'
        
        # prepare the checkpoint
        if not os.path.exists(ckpt_path):
            ckpt_url='https://github.com/ozanciga/self-supervised-histopathology/releases/download/tenpercent/tenpercent_resnet18.ckpt'
            wget.download(ckpt_url, out=ckpt_dir)
            
        state = torch.load(ckpt_path)
        state_dict = state['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
        
        model_dict = resnet.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        if state_dict == {}:
            print('No weight could be loaded..')
        model_dict.update(state_dict)
        resnet.load_state_dict(model_dict)
        resnet.fc = nn.Identity()

        return resnet


class TRIPLEX(nn.Module):
    """Model class for TRIPLEX
    """
    def __init__(self, 
                num_genes=250,
                emb_dim=512,
                depth1=2,
                depth2=2,
                depth3=2,
                num_heads1=8,
                num_heads2=8,
                num_heads3=8,
                mlp_ratio1=2.0,
                mlp_ratio2=2.0,
                mlp_ratio3=2.0,
                dropout1=0.1,
                dropout2=0.1,
                dropout3=0.1,
                kernel_size=3,
                res_neighbor=(5,5),
                pos_layer='APEG',
                max_batch_size=1024,
                selector_mode='none',
                selector_dim=64,
                selector_temperature=1.0,
                selector_budget_loss_weight=0.0,
                selector_use_similarity_prior=False,
                selector_similarity_prior_weight=1.0,
                selector_learned_score_weight=1.0,
                num_selected_patches=20,
                selector_ckpt_path=None):
        """TRIPLEX model 

        Args:
            num_genes (int): Number of genes to predict.
            emb_dim (int): Embedding dimension for images. Defaults to 512.
            depth1 (int): Depth of FusionEncoder. Defaults to 2.
            depth2 (int): Depth of GlobalEncoder. Defaults to 2.
            depth3 (int): Depth of NeighborEncoder. Defaults to 2.
            num_heads1 (int): Number of heads for FusionEncoder. Defaults to 8.
            num_heads2 (int): Number of heads for GlobalEncoder. Defaults to 8.
            num_heads3 (int): Number of heads for NeighborEncoder. Defaults to 8.
            mlp_ratio1 (float): mlp_ratio (MLP dimension/emb_dim) for FusionEncoder. Defaults to 2.0.
            mlp_ratio2 (float): mlp_ratio (MLP dimension/emb_dim) for GlobalEncoder. Defaults to 2.0.
            mlp_ratio3 (float): mlp_ratio (MLP dimension/emb_dim) for NeighborEncoder. Defaults to 2.0.
            dropout1 (float): Dropout rate for FusionEncoder. Defaults to 0.1.
            dropout2 (float): Dropout rate for GlobalEncoder. Defaults to 0.1.
            dropout3 (float): Dropout rate for NeighborEncoder. Defaults to 0.1.
            kernel_size (int): Kernel size of convolution layer in PEGH. Defaults to 3.
            res_neighbor (tuple): Resolution of neighbor embeddings. Defaults to (5,5).
            max_batch_size (int): Maximum batch size for inference. Defaults to 1024.
        """
        
        super().__init__()
        
        self.alpha = 0.3
        self.emb_dim = emb_dim
        self.max_batch_size = max_batch_size
        self.selector_mode = selector_mode
        self.selector_budget_loss_weight = selector_budget_loss_weight

        valid_selector_modes = {
            'none', 'soft', 'straight_through', 'frozen_hard'
        }
        if selector_mode not in valid_selector_modes:
            raise ValueError(
                f"selector_mode must be one of {sorted(valid_selector_modes)}, "
                f"but got {selector_mode}"
            )

        self.neighbor_selector = None
        if selector_mode != 'none':
            self.neighbor_selector = TargetConditionedSelector(
                emb_dim=emb_dim,
                selector_dim=selector_dim,
                num_selected_patches=num_selected_patches,
                temperature=selector_temperature,
                use_similarity_prior=selector_use_similarity_prior,
                similarity_prior_weight=selector_similarity_prior_weight,
                learned_score_weight=selector_learned_score_weight,
            )
            if selector_mode == 'frozen_hard':
                if not selector_ckpt_path:
                    raise ValueError(
                        "selector_ckpt_path is required when "
                        "selector_mode='frozen_hard'"
                    )
                load_selector_state(self.neighbor_selector, selector_ckpt_path)
                freeze_selector(self.neighbor_selector)
    
        # Target Encoder
        resnet18 = load_model_weights("tenpercent_resnet18.ckpt")
        module=list(resnet18.children())[:-2]
        self.target_encoder = nn.Sequential(*module)
        self.fc_target = nn.Linear(emb_dim, num_genes)
        
        self.target_linear = nn.Linear(512, emb_dim)

        # Neighbor Encoder
        self.neighbor_encoder = NeighborEncoder(emb_dim, 
                                                depth3, 
                                                num_heads3, 
                                                int(emb_dim*mlp_ratio3), 
                                                dropout = dropout3, 
                                                resolution=res_neighbor)
        self.fc_neighbor = nn.Linear(emb_dim, num_genes)

        # Global Encoder        
        self.global_encoder = GlobalEncoder(emb_dim, 
                                            depth2, 
                                            num_heads2, 
                                            int(emb_dim*mlp_ratio2), 
                                            dropout2, 
                                            kernel_size,
                                            pos_layer)
        self.fc_global = nn.Linear(emb_dim, num_genes)
        
        # Fusion Layer
        self.fusion_encoder = FusionEncoder(emb_dim, 
                                            depth1, 
                                            num_heads1, 
                                            int(emb_dim*mlp_ratio1), 
                                            dropout1)    
        
        self.fc = nn.Linear(emb_dim, num_genes)
        
    def forward(self,
                img, 
                mask, 
                neighbor_emb, 
                position=None, 
                global_emb=None, 
                pid=None, 
                sid=None, 
                neighbor_pool_mask=None,
                **kwargs):
        
        if 'dataset' in kwargs:
            # Training
            return self._process_training_batch(
                img,
                mask,
                neighbor_emb,
                pid,
                sid,
                kwargs['dataset'],
                kwargs['label'],
                neighbor_pool_mask,
            )
        else:
            # Inference 
            return self._process_inference_batch(img, mask, neighbor_emb, position, global_emb, sid)
            
    def _process_training_batch(
            self,
            img,
            mask,
            neighbor_emb,
            pid,
            sid,
            dataset,
            label,
            neighbor_pool_mask=None):
        global_emb, position = self.retrieve_global_emb(pid, dataset)
        
        (
            fusion_token,
            target_token,
            neighbor_token,
            global_token,
            effective_pool_mask,
            selector_budget_loss,
        ) = self._encode_all(
            img,
            mask,
            neighbor_emb,
            position,
            global_emb,
            pid,
            sid,
            neighbor_pool_mask,
        )
        return self._get_outputs(
            fusion_token,
            target_token,
            neighbor_token,
            global_token,
            label,
            effective_pool_mask,
            selector_budget_loss,
        )

    def _process_inference_batch(self, img, mask, neighbor_emb, position, global_emb, sid=None):
        if sid is None and img.shape[0] > self.max_batch_size:
            imgs = img.split(self.max_batch_size, dim=0)
            neighbor_embs = neighbor_emb.split(self.max_batch_size, dim=0)
            masks = mask.split(self.max_batch_size, dim=0)
            sid = torch.arange(img.shape[0]).to(img.device)
            sids = sid.split(self.max_batch_size, dim=0)
            
            pred = [self.fc(self._encode_all(img, mask, neighbor_emb, position, global_emb, sid=sid)[0]) \
                for img, neighbor_emb, mask, sid in zip(imgs, neighbor_embs, masks, sids)]
            pred = torch.clamp(torch.cat(pred, dim=0), 0) 
            
            return {'logits': pred}    
        else:
            fusion_token, _, _, _, _, _ = self._encode_all(
                img, mask, neighbor_emb, position, global_emb, sid=sid
            )
            pred = torch.clamp(self.fc(fusion_token), 0)
        return {'logits': pred}
    
    def _encode_all(
            self,
            img,
            mask,
            neighbor_emb,
            position,
            global_emb,
            pid=None,
            sid=None,
            neighbor_pool_mask=None):
        selector_gate = None
        selector_budget_loss = neighbor_emb.sum() * 0.0
        effective_pool_mask = neighbor_pool_mask

        if self.neighbor_selector is not None:
            selector_result = self.neighbor_selector(
                neighbor_emb,
                mask,
                self.selector_mode,
            )
            selector_gate = selector_result['gate']
            selector_budget_loss = selector_result['budget_loss']
            effective_pool_mask = selector_gate

        target_token = self.encode_target(img)
        neighbor_token = self.neighbor_encoder(
            neighbor_emb,
            mask,
            gate=selector_gate,
        )
        global_token = self.encode_global(global_emb, position, pid, sid)
        
        fusion_token = self.fusion_encoder(
            target_token,
            neighbor_token,
            global_token,
            mask=mask,
            gate=selector_gate,
        )
        
        return (
            fusion_token,
            target_token,
            neighbor_token,
            global_token,
            effective_pool_mask,
            selector_budget_loss,
        )
    
    def encode_target(self, img):
        # Target tokens
        target_token = self.target_encoder(img) # B x 512 x 7 x 7
        B, dim, w, h = target_token.shape
        target_token = rearrange(target_token, 'b d h w -> b (h w) d', d = dim, w=w, h=h)
        target_token = self.target_linear(target_token)
        
        return target_token
        
    def encode_global(self, global_emb, position, pid=None, sid=None):
        # Global tokens
        if isinstance(global_emb, dict):
            global_token = torch.zeros((sid.shape[0], self.emb_dim)).to(sid.device)
            for _id, x_g in global_emb.items():
                batch_idx = pid == _id
                pos = position[_id]
                # x_cond_encoded = self.encode_cond(x_g, pos[id_]) # N x D
                g_token = self.global_encoder(x_g, pos).squeeze()  # N x 512
                global_token[batch_idx] = g_token[sid[batch_idx]] # B x D
        else:
            global_token = self.global_encoder(global_emb, position).squeeze()  # N x 512
            if sid is not None:
                global_token = global_token[sid]
                
        return global_token
        
    def _get_outputs(
            self,
            fusion_token,
            target_token,
            neighbor_token,
            global_token,
            label,
            neighbor_pool_mask=None,
            selector_budget_loss=None):
        output = torch.clamp(self.fc(fusion_token), 0) # B x num_genes
        out_target = torch.clamp(self.fc_target(target_token.mean(1)), 0) # B x num_genes

        if neighbor_pool_mask is None:
            # Preserve the original TRIPLEX behavior for the baseline config.
            neighbor_pooled = neighbor_token.mean(1)
        else:
            pool_mask = neighbor_pool_mask.to(
                device=neighbor_token.device,
                dtype=neighbor_token.dtype,
            ).unsqueeze(-1)
            neighbor_pooled = (neighbor_token * pool_mask).sum(dim=1)
            neighbor_pooled = neighbor_pooled / pool_mask.sum(dim=1).clamp_min(1.0)

        out_neighbor = torch.clamp(self.fc_neighbor(neighbor_pooled), 0) # B x num_genes
        out_global = torch.clamp(self.fc_global(global_token), 0) # B x num_genes
        
        preds = (output, out_target, out_neighbor, out_global)
        
        loss = self.calculate_loss(preds, label)
        if selector_budget_loss is not None:
            loss = loss + (
                self.selector_budget_loss_weight * selector_budget_loss
            )
        
        return {
            'loss': loss,
            'logits': output,
            'selector_budget_loss': selector_budget_loss,
        }

    def train(self, mode=True):
        super().train(mode)
        if self.selector_mode == 'frozen_hard':
            self.neighbor_selector.eval()
        return self
        
    def calculate_loss(self, preds, label):
        
        loss = F.mse_loss(preds[0], label)                       # Supervised loss for Fusion
        
        for i in range(1, len(preds)):
            loss += F.mse_loss(preds[i], label) * (1-self.alpha) # Supervised loss
            loss += F.mse_loss(preds[0], preds[i]) * self.alpha  # Distillation loss
    
        return loss
    
    def retrieve_global_emb(self, pid, dataset):
        device = pid.device
        unique_pid = pid.unique()
        
        global_emb = {}
        pos = {}
        for pid in unique_pid:
            pid = int(pid)
            _id = dataset.int2id[pid]
            
            global_emb[pid] = dataset.global_embs[_id].clone().to(device).unsqueeze(0)
            pos[pid] = dataset.pos_dict[_id].clone().to(device)
        
        return global_emb, pos
