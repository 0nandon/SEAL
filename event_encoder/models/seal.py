
from functools import partial
from typing import Tuple, List, Optional, Type
import math
from operator import mul
from functools import reduce

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torchvision
from models.basemodule import BaseModule
from models.pixel_decoder import get_decoder

from utils.loss import loss_func

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from segment_anything import sam_evimg_model_registry, SamPredictor
from segment_anything.utils.mask_postprocess import *
from pointnet2.pointnet2_utils import furthest_point_sample

class SEAL(BaseModule):
    def __init__(self, ctx):
        super().__init__(init_cfg=None)
        
        self.ctx = ctx
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_levels = len(ctx.mask_levels)
        self.mask_levels = ctx.mask_levels
        self.roi = ctx.roi
        self.use_decoder = ctx.use_decoder
        self.semantic_width = ctx.semantic_width
        
        self.num_event_modality_prompt = ctx.neck.num_event_modality_prompt
        self.event_prompt_in_cross_attn = ctx.neck.event_prompt_in_cross_attn

        self.roi_align = torchvision.ops.RoIAlign(
            output_size= ctx.roi_pool.output_size,
            spatial_scale=ctx.roi_pool.spatial_scale,
            sampling_ratio=ctx.roi_pool.sampling_ratio, 
            aligned=ctx.roi_pool.aligned
        )
        
        self.eventsam = sam_evimg_model_registry[ctx.backbone.model_type](
            input_signal=ctx.backbone.input_signal,
            encoder_checkpoint=ctx.backbone.encoder_checkpoint,
            decoder_checkpoint=ctx.backbone.decoder_checkpoint
        )
        self.predictor = SamPredictor(self.eventsam)
        
        self.neck = get_neck(ctx.neck)
        self.in_proj = nn.Linear(ctx.in_proj.in_channels, ctx.aggregate.embed_channels)
        
        self.pe_layer = PositionEmbeddingRandom(ctx.aggregate.embed_channels // 2)
        self.fuse = CrossBlock(dim=ctx.aggregate.embed_channels)

        if ctx.aggregate.embed_channels != ctx.output_dim:
            self.out_proj = nn.Linear(ctx.aggregate.embed_channels, ctx.output_dim)
        else:
            self.out_proj = None

        if self.use_decoder:
            self.pixel_decoder = get_decoder(ctx.decoder)

        self.mask_tokens = None
        
        for name, param in self.eventsam.named_parameters():
            param.requires_grad = False
            
    def compute_batched_mask(self, fusions):
        max_size = 0
        for fusion in fusions:
            for lvl in self.mask_levels:
                max_size = fusion[lvl].shape[0] if fusion[lvl].shape[0] > max_size else max_size
        
        if self.event_prompt_in_cross_attn:
            max_size += self.num_event_modality_prompt
            
        semantic_key_mask = []
        semantic_key_idx = []
        key = []  
        for fusion in fusions:
            for lvl in self.mask_levels:
                key_size = fusion[lvl].shape[0]

                idx = torch.zeros(
                    max_size,
                    dtype=torch.long,
                    device=fusion[lvl].device,
                )

                midx = torch.ones(
                    max_size,
                    dtype=torch.bool,
                    device=fusion[lvl].device,
                )

                if self.event_prompt_in_cross_attn:
                    idx[self.num_event_modality_prompt:self.num_event_modality_prompt+key_size] = torch.arange(
                        key_size, device=fusion[lvl].device
                    )
                    midx[:key_size + self.num_event_modality_prompt] = False
                else:
                    idx[:key_size] = torch.arange(
                        key_size, device=fusion[lvl].device
                    )
                    midx[:key_size] = False
                
                key.append(fusion[lvl])
                semantic_key_mask.append(midx)
                semantic_key_idx.append(idx)
        
        batched_text_key = torch.stack(
            [
                key[k][semantic_key_idx[k], :]
                for k in range(len(key))
            ]
        )
        batched_text_key_mask = torch.stack(semantic_key_mask)
        batched_text_key_mask = batched_text_key_mask[..., None].repeat(1, 1, self.ctx.neck.input_size[0] ** 2)
        batched_text_key_mask = batched_text_key_mask.repeat_interleave(
            self.ctx.neck.num_heads, dim=0
        ).permute(0, 2, 1)

        return batched_text_key.float(), batched_text_key_mask.float()
    
    def get_batch_roi_bboxes(self, masks: torch.Tensor):
        """
        Compute ROI bounding boxes for a batch of binary segmentation masks.
    
        Args:
            masks (torch.Tensor): Tensor of shape (N, H, W) with binary values (0 and 1).
        
        Returns:
            roi_boxes (torch.Tensor): Tensor of shape (num_valid_rois, 5) in format
                                  [batch_index, x1, y1, x2, y2].
            valid (torch.BoolTensor): Boolean tensor of shape (N,) indicating which masks have at least one foreground pixel.
        """
        N, H, W = masks.shape
        device = masks.device

        rows = torch.arange(H, device=device).view(1, H, 1).expand(N, H, W)
        cols = torch.arange(W, device=device).view(1, 1, W).expand(N, H, W)
    
        masked_rows = torch.where(masks.bool(), rows, torch.tensor(H, device=device))
        masked_cols = torch.where(masks.bool(), cols, torch.tensor(W, device=device))
    
        min_rows = masked_rows.view(N, -1).min(dim=1).values
        min_cols = masked_cols.view(N, -1).min(dim=1).values

        masked_rows_max = torch.where(masks.bool(), rows, torch.tensor(-1, device=device))
        masked_cols_max = torch.where(masks.bool(), cols, torch.tensor(-1, device=device))
    
        max_rows = masked_rows_max.view(N, -1).max(dim=1).values
        max_cols = masked_cols_max.view(N, -1).max(dim=1).values
        valid = max_rows >= 0
    
        batch_indices = torch.zeros(N, device=device)
        roi_boxes = torch.stack([batch_indices.float(), 
                                min_cols.float(), 
                                min_rows.float(), 
                                max_cols.float(), 
                                max_rows.float()], dim=1)
    
        roi_boxes = roi_boxes[valid]
        return roi_boxes, valid

    def forward(self, x, x_rgb, fusion_short=None, fusion_long=None, fusion_img=None, weight=None, mask_gt=None, mask_x2_gt=None):
        if self.ctx.neck.multimodal_fusion:
            batched_fusion, batched_fusion_key_mask = self.compute_batched_mask(fusion_long)
        else:
            batched_fusion, batched_fusion_key_mask = None, None

        with torch.no_grad():
            x = self.eventsam.preprocess(x, evimg=True)
            feat_backbone, feat = self.eventsam.image_encoder(x)
            
            granularity_tokens = []
            for feat_backbone_per_image, mask_gt_per_image in zip(feat_backbone, mask_x2_gt):
                granularity_tokens.append({})
                for k in mask_gt_per_image.keys():
                    bbox, _ = self.get_batch_roi_bboxes(mask_gt_per_image[k])
                    bbox = bbox[:, 1:]
                
                    sparse_embeddings, dense_embeddings = self.eventsam.prompt_encoder(
                        points=None,
                        boxes=bbox,
                        masks=None
                    )
                
                    _, scores, tokens = self.eventsam.mask_decoder(
                        image_embeddings=feat_backbone_per_image[None, :, :, :],
                        image_pe=self.eventsam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=True,
                        return_tokens=True
                    )
                    
                    select_idx = scores.argmax(dim=-1)
                    tokens = tokens[torch.arange(tokens.shape[0]), select_idx]
                    granularity_tokens[-1][k] = tokens

        feat = feat[:, None, :, :, :].repeat(1, self.num_levels, 1, 1, 1)
        B, L, H, W, D = feat.shape
        
        if self.mask_tokens is not None:
            mask_tokens = torch.stack([self.mask_tokens[i].weight for i in range(self.num_levels)], dim=0)
            mask_tokens = mask_tokens.reshape(L, H, W, D)
            mask_tokens = mask_tokens[None, :, :, :, :].repeat(B, 1, 1, 1, 1)
            feat = feat + mask_tokens

        feat = feat.reshape(B * L, H, W, D)
        
        if self.ctx.neck.multimodal_fusion:
            feat = self.neck(feat, batched_fusion, batched_fusion_key_mask).permute(0, 3, 1, 2)
        else:
            feat = self.neck(feat).permute(0, 3, 1, 2)
        
        if self.use_decoder:
            feat = self.pixel_decoder(feat)
            feat = feat.reshape(B, L, self.ctx.neck.out_channels, H * self.ctx.decoder.upscale, W * self.ctx.decoder.upscale)
        else:
            feat = feat.reshape(B, L, self.ctx.neck.out_channels, H, W)
        
        loss = {}
        loss['total'] = []
        for lvl in self.mask_levels:
            loss[lvl] = []

        for i, (feat_per_image, roi_gt_per_image) in enumerate(zip(feat, mask_gt)):
            for j, k in enumerate(roi_gt_per_image.keys()):
                if k not in self.mask_levels:
                    continue
                
                feat_per_lvl = feat_per_image[j].permute(1, 2, 0)
                
                if self.use_decoder:
                    roi_gt_per_image_ = torch.nn.functional.interpolate(
                        roi_gt_per_image[k][None, :, :, :].float(),
                        size=128,
                        mode='bilinear'
                    )[0].bool()
                else:
                    roi_gt_per_image_ = torch.nn.functional.adaptive_max_pool2d(
                        roi_gt_per_image[k][None, :, :, :].float(), 
                        self.ctx.neck.input_size[0] if not self.use_decoder else self.ctx.neck.input_size[0] * self.ctx.decoder.upscale
                    ) == 1
                    roi_gt_per_image_ = roi_gt_per_image_[0]
                
                if self.roi:
                    roi_boxes, _ = self.get_batch_roi_bboxes(roi_gt_per_image_)

                    roi_features = self.roi_align(
                        feat_per_lvl.permute(2, 0, 1)[None, :, :, :],
                        roi_boxes,
                    ).mean(dim=-1).mean(dim=-1)
                    
                    roi_features = torch.cat([roi_features, granularity_tokens[i][k]], dim=-1)
                    roi_features = self.in_proj(roi_features)
                    
                    B, _ = roi_features.shape
                    batched_fusion_key_mask = (~roi_gt_per_image_).reshape(B, -1)
                    batched_fusion_key_mask = batched_fusion_key_mask[None, :, :].repeat(self.ctx.neck.num_heads, 1, 1)

                    if self.use_decoder:
                        pos_coord_size = [
                            self.ctx.neck.input_size[0] * self.ctx.decoder.upscale, 
                            self.ctx.neck.input_size[1] * self.ctx.decoder.upscale
                        ]
                    else:
                        pos_coord_size = self.ctx.neck.input_size

                    roi_features = self.fuse(
                        x=roi_features[None, :, :],
                        batched_fusion=feat_per_lvl.reshape(-1, self.ctx.aggregate.embed_channels)[None, :, :],
                        batched_fusion_key_mask=batched_fusion_key_mask,
                        image_pe = self.pe_layer(pos_coord_size).permute(1, 2, 0).reshape(-1, self.ctx.position.embed_channels)[None, :, :]
                    )

                    if self.out_proj is not None:
                        roi_features = self.out_proj(roi_features)
                    
                    mask_feat_lst = torch.nn.functional.normalize(roi_features, p=2.0, dim=-1)
                
                loss['total'].append(
                    (loss_func(mask_feat_lst, fusion_short[i][k], self.ctx.loss, weight=weight[i][k]) + \
                    loss_func(mask_feat_lst, fusion_img[i][k], self.ctx.loss, weight=weight[i][k]))
                )
                loss[k].append(loss['total'][-1])
        
        loss['total'] = torch.sum(torch.stack(loss['total']))
        for lvl in self.mask_levels:
            loss[lvl] = torch.sum(torch.stack(loss[lvl]))

        return loss
    
    def generate_prompt(self, mask, type="point", point_num=3, sample="random"):
        if type == "point":
            if sample == "random":
                coords_x, coords_y = np.where(mask)
                random_idx = np.random.choice(coords_x.shape[0], point_num)
                points = np.vstack([coords_y[random_idx], coords_x[random_idx]]).transpose()
                labels = np.ones((points.shape[0]))
            elif sample == "furthest":
                coords_x, coords_y = np.where(mask)
                points = np.vstack([coords_y, coords_x]).transpose()
                points_3D = np.concatenate([points, np.zeros((points.shape[0], 1))], axis=-1)
                idx = furthest_point_sample(
                    torch.from_numpy(points_3D).cuda().float()[None, :, :].contiguous(),
                    point_num,
                ).detach().cpu().numpy()
                points = points[idx[0]]
                labels = np.ones((points.shape[0]))
            return points, labels
        elif type == "box":
            indices = np.argwhere(mask)
            if indices.size == 0:
                return None

            y_min, x_min = indices.min(axis=0)
            y_max, x_max = indices.max(axis=0)
            
            return np.array([x_min, y_min, x_max, y_max])
    
    def set_image(self, evimg):
        with torch.no_grad():
            self.predictor.set_image(evimg)
    
    def predict_mask(self, input_points=None, input_labels=None, input_box=None):
        masks, scores, logits, tokens = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_box,
            multimask_output=True,
            return_tokens=True
        )

        masks = torch.from_numpy(masks).to(self.device)
        scores = torch.from_numpy(scores).to(self.device)
        
        mask_filter = []
        for mask in masks:
            if mask.sum() == 0:
                mask_filter.append(False)
            else:
                mask_filter.append(True)
        mask_filter = torch.tensor(mask_filter)
            
        scores = scores[mask_filter]
        masks = masks[mask_filter]
        tokens = tokens[mask_filter]
        
        return masks, scores, tokens
    
    def predict_label(self, masks, tokens, fusion=None):
        feat = self.predictor.features_med

        B, H, W, D = feat.shape
        fusion = fusion[None, :, :].repeat(B, 1, 1)
        if self.ctx.neck.multimodal_fusion:
            feat = self.neck(feat, fusion).permute(0, 3, 1, 2)
        else:
            feat = self.neck(feat).permute(0, 3, 1, 2)

        if self.use_decoder:
            feat = self.pixel_decoder(feat)

        feat = feat[0].permute(1, 2, 0)
        
        if self.use_decoder:
            masks = torch.nn.functional.interpolate(
                masks[None, :, :, :].float(),
                size=128,
                mode='bilinear'
            )[0].bool()
        else:   
            masks = (
                torch.nn.functional.adaptive_max_pool2d(
                    masks[None, :, :, :].float(), 
                    self.ctx.neck.input_size[0] if not self.use_decoder else self.ctx.neck.input_size[0] * self.ctx.decoder.upscale
                ) == 1
            )[0]
        
        roi_boxes, _ = self.get_batch_roi_bboxes(masks)
        roi_features = self.roi_align(
            feat[None, :, :, :].permute(0, 3, 1, 2),
            roi_boxes,
        ).mean(dim=-1).mean(dim=-1)

        roi_features = torch.cat([roi_features, tokens], dim=-1)
        roi_features = self.in_proj(roi_features)

        B, _ = roi_features.shape
        batched_fusion_key_mask = (~masks).reshape(B, -1)
        batched_fusion_key_mask = batched_fusion_key_mask[None, :, :].repeat(self.ctx.neck.num_heads, 1, 1)

        if self.use_decoder:
            pos_coord_size = [
                self.ctx.neck.input_size[0] * self.ctx.decoder.upscale, 
                self.ctx.neck.input_size[1] * self.ctx.decoder.upscale
            ]
        else:
            pos_coord_size = self.ctx.neck.input_size

        roi_features = self.fuse(
            x=roi_features[None, :, :],
            batched_fusion=feat.reshape(-1, self.ctx.aggregate.embed_channels)[None, :, :],
            batched_fusion_key_mask=batched_fusion_key_mask,
            image_pe = self.pe_layer(pos_coord_size).permute(1, 2, 0).reshape(-1, self.ctx.position.embed_channels)[None, :, :]
        )[0]
        
        if self.out_proj is not None:
            roi_features = self.out_proj(roi_features)
        
        roi_features_norm = torch.nn.functional.normalize(roi_features, p=2.0, dim=-1)
        costmap = (roi_features_norm @ fusion[0].t() * 100).softmax(dim=-1)
        scores, labels = costmap.max(dim=-1)
        
        return labels, scores, costmap
    
    def predict(self, input_points=None, input_labels=None, input_box=None, fusion=None):
        
        masks, scores, tokens = self.predict_mask(
            input_points=input_points, 
            input_labels=input_labels,
            input_box=input_box
        )
        
        if len(masks) == 0:
            return None, None, None, -1

        labels, class_scores, costmap = self.predict_label(masks, tokens, fusion)

        total_scores = scores * class_scores
        idx = total_scores.argmax()

        return masks, labels, total_scores, idx
    
# =============================================

def get_neck(ctx):
    if ctx.name == "MultiLayerSAMTransformerNeck":
        return MultiLayerSAMTransformerNeck(
            input_size=ctx.input_size,
            in_channels=ctx.in_channels,
            embed_channels=ctx.embed_channels,
            out_channels=ctx.out_channels,
            depth=ctx.depth,
            global_attn_indexes=ctx.global_attn_indexes,
            window_size=ctx.window_size,
            num_heads=ctx.num_heads,
            mlp_ratio=ctx.mlp_ratio,
            use_rel_pos=ctx.use_rel_pos,
            rel_pos_zero_init=ctx.rel_pos_zero_init,
            multimodal_fusion=ctx.multimodal_fusion,
            num_event_modality_prompt=ctx.num_event_modality_prompt,
            event_prompt_in_cross_attn=ctx.event_prompt_in_cross_attn
        )


class MultiLayerSAMTransformerNeck(BaseModule):
    STRIDE = 16

    def __init__(
            self,
            input_size: Tuple[int, int],
            in_channels: List[int],
            embed_channels: int,
            out_channels: int,
            depth: int,
            global_attn_indexes: List[int],
            window_size: int,
            num_heads: int,
            mlp_ratio: int,
            use_rel_pos: bool,
            rel_pos_zero_init: bool,
            multimodal_fusion: bool,
            num_event_modality_prompt: int,
            event_prompt_in_cross_attn: bool
    ) -> None:
        super().__init__(init_cfg=None)

        self.transformer_size = input_size
        self.multimodal_fusion = multimodal_fusion

        self.blocks = nn.ModuleList()
        for i in range(depth):
            if self.multimodal_fusion:
                block = FusionBlock(
                    dim=embed_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                    act_layer=nn.GELU,
                    use_rel_pos=use_rel_pos,
                    rel_pos_zero_init=rel_pos_zero_init,
                    window_size=window_size if i not in global_attn_indexes else 0,
                    input_size=self.transformer_size,
                    num_event_modality_prompt=num_event_modality_prompt,
                    event_prompt_in_cross_attn=event_prompt_in_cross_attn
                )
                self.blocks.append(block)
            else:
                block = Block(
                    dim=embed_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                    act_layer=nn.GELU,
                    use_rel_pos=use_rel_pos,
                    rel_pos_zero_init=rel_pos_zero_init,
                    window_size=window_size if i not in global_attn_indexes else 0,
                    input_size=self.transformer_size,
                )
                self.blocks.append(block)

        self.in_proj = None
        self.in_proj_fusion = None
        self.out_proj = None
        self.out_proj_fusion = None
        
        if in_channels != embed_channels:
            self.in_proj = nn.Linear(in_channels, embed_channels)
            if self.multimodal_fusion:
                self.in_proj_fusion = nn.Linear(in_channels, embed_channels)
        if out_channels != embed_channels:
            self.out_proj = nn.Linear(embed_channels, out_channels)
            if self.multimodal_fusion:
                self.out_proj_fusion = nn.Linear(embed_channels, out_channels)

    def forward(self, x, batched_fusion=None, batched_fusion_key_mask=None) -> Tensor:
        if self.in_proj is not None:
            x = self.in_proj(x)
        if self.in_proj_fusion is not None:
            batched_fusion = self.in_proj_fusion(batched_fusion)
            
        for block in self.blocks:
            if self.multimodal_fusion:
                x = block(x, batched_fusion, batched_fusion_key_mask)
            else:
                x = block(x)
            
        if self.out_proj is not None:
            x = self.out_proj(x)
        if self.out_proj_fusion is not None:
            batched_fusion = self.out_proj_fusion(batched_fusion)
        return x


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
            
        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class FusionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
        num_event_modality_prompt: int = 0,
        event_prompt_in_cross_attn: bool = False
    ) -> None:
        super().__init__()
        
        self.num_event_modality_prompt = num_event_modality_prompt
        self.event_prompt_in_cross_attn = event_prompt_in_cross_attn
        
        if num_event_modality_prompt > 0:
            self.event_prompt = InsertEventPrompt(
                feature_dim=dim,
                num_event_modality_prompt=num_event_modality_prompt,
                patch_size = input_size if window_size == 0 else (window_size, window_size)
            )
        else:
            self.event_prompt = None
            
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
            num_event_modality_prompt = num_event_modality_prompt if not event_prompt_in_cross_attn else 0
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=num_heads,
            batch_first=True
        )

        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.window_size = window_size

    
    def forward(
        self, 
        x: torch.Tensor, 
        batched_fusion: torch.Tensor, 
        batched_fusion_key_mask: torch.Tensor
    ) -> torch.Tensor:        
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size) 
           
        if self.event_prompt is not None and not self.event_prompt_in_cross_attn: 
            B, H_, W_, D = x.shape
            x = x.reshape(B, -1, D)
            x = self.event_prompt(x, x.shape[0]).unsqueeze(1)
            x = self.attn(x)
            x = x[:, 0, self.event_prompt.num_event_modality_prompt:, :].reshape(B, H_, W_, D)
        else:
            x = self.attn(x)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        
        B, H, W, C = x.shape
        x = x.reshape(B, -1, C)
        
        if self.event_prompt_in_cross_attn:
            event_prompt = self.event_prompt.event_modality_prompts.unsqueeze(0).repeat(B, 1, 1)
            
            if self.eval:
                batched_fusion = torch.cat([event_prompt, batched_fusion], dim=1)
            else:
                batched_fusion[:, :self.num_event_modality_prompt, :] = event_prompt

        x = self.cross_attn(
            x, 
            batched_fusion, 
            batched_fusion,
            attn_mask = batched_fusion_key_mask
        )[0]

        x = x.reshape(B, H, W, C)

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class CrossBlock(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=num_heads,
            batch_first=True
        )

        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        batched_fusion: torch.Tensor, 
        batched_fusion_key_mask: torch.Tensor,
        image_pe: torch.Tensor
    ) -> torch.Tensor:        
        shortcut = x
        x = self.norm1(x)

        x = self.cross_attn(
            x, 
            batched_fusion + image_pe if image_pe is not None else batched_fusion, 
            batched_fusion,
            attn_mask = batched_fusion_key_mask
        )[0]

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
        num_event_modality_prompt: int = 0
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
            
        self.input_size = input_size
        self.num_event_modality_prompt = num_event_modality_prompt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        
        attn = (q * self.scale) @ k.transpose(-2, -1)
        
        if self.use_rel_pos:
            if self.num_event_modality_prompt == 0:
                attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
            else:
                attn = add_decomposed_rel_pos_w_prompt(
                    attn, 
                    q, 
                    self.rel_pos_h, 
                    self.rel_pos_w, 
                    self.input_size, 
                    self.input_size,
                    self.num_event_modality_prompt
                )
                
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))
    

class MLPBlockFlex(nn.Module):
    def __init__(
        self,
        input_dim: int,
        mlp_dim: int,
        output_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(input_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, output_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class InsertEventPrompt(nn.Module):
    def __init__(
            self,
            feature_dim,
            num_event_modality_prompt,
            patch_size,
    ):
        super().__init__()
        
        feature_dim = feature_dim
        self.num_event_modality_prompt = num_event_modality_prompt
        self.event_modality_prompts = nn.Parameter(torch.zeros(num_event_modality_prompt, feature_dim)) 
        self._initialize_event_modality_prompts(patch_size, feature_dim)

    def _initialize_event_modality_prompts(self, patch_size, prompt_dim):
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))
        nn.init.uniform_(self.event_modality_prompts.data, -val, val)

    def forward(self, x, B):
        device = x.device
        event_modality_prompts = self.event_modality_prompts.repeat(B, 1, 1).to(device)
        x = torch.cat([event_modality_prompts, x], dim=1)
        return x
    

def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


def add_decomposed_rel_pos_w_prompt(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
    num_event_modality_prompt: int
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)
    
    B, _, dim = q.shape
    q = q[:, num_event_modality_prompt:, :]
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
    
    attn_img = attn[:, -q_h * q_w:, -q_h * q_w:]
    attn_img = (
        attn_img.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)
    attn[:, -q_h * q_w:, -q_h * q_w:] = attn_img
    
    return attn
