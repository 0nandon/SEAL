
from functools import partial
from typing import Tuple, List, Optional, Type
import math
from operator import mul
from functools import reduce

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torchvision

import clip
from models.basemodule import BaseModule
from models.resnet import ResNet
from models.e2vid.utils.loading_utils import load_model
from models.e2vid.image_reconstructor import ImageReconstructor
from models.e2vid.config.settings import Settings
from models.e2vid.model.submodules import ResidualBlock
from models.snn.snn_vgg import SNN_VGG
from utils.loss import loss_func

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from segment_anything import sam_evimg_model_registry, SamPredictor
from segment_anything.utils.mask_postprocess import *
from pointnet2.pointnet2_utils import furthest_point_sample
from PIL import Image

from utils.timer import CudaTimer

class BASELINE(BaseModule):
    def __init__(self, ctx):
        super().__init__(init_cfg=None)
        
        self.ctx = ctx
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.type = ctx.type
        
        self.eventsam = sam_evimg_model_registry[ctx.backbone.model_type](
            input_signal=ctx.backbone.input_signal,
            encoder_checkpoint=ctx.backbone.encoder_checkpoint,
            decoder_checkpoint=ctx.backbone.decoder_checkpoint
        )
        self.predictor = SamPredictor(self.eventsam)

        for name, param in self.eventsam.named_parameters():
            param.requires_grad = False
        
        if self.type == "image_crop":
            clip_model, clip_preprocess = clip.load(ctx.clip_model, self.device)
            self.clip = clip_model
            self.clip_preprocess = clip_preprocess
        elif self.type == "frame2recon":
            self.neck = ResNet(
                layers=ctx.neck.layers,
                output_dim=ctx.neck.embed_dim,
                heads=ctx.neck.width * 32 // ctx.neck.head_width,
                image_size=ctx.neck.image_size,
                width=ctx.neck.width
            )
            
            for name, param in self.neck.named_parameters():
                param.requires_grad = True
                
            self.roi_align = torchvision.ops.RoIAlign(
                output_size= ctx.roi_pool.output_size,
                spatial_scale=ctx.roi_pool.spatial_scale,
                sampling_ratio=ctx.roi_pool.sampling_ratio, 
                aligned=ctx.roi_pool.aligned
            )
            
            self.out_proj = nn.Linear(ctx.neck.embed_dim, ctx.output_dim)
        elif self.type == "frame2voxel":
            self.e2vid, _ = load_model(ctx.neck.path, pretrain=True)
            self.neck = ImageReconstructor(
                self.e2vid.to(self.device), 
                ctx.neck.voxel_height, 
                ctx.neck.voxel_width,
                ctx.neck.voxel_channel, 
                self.device
            )
            
            for name, param in self.e2vid.named_parameters():
                param.requires_grad = False
            
            self.resblocks = nn.ModuleList()
            for i in range(ctx.neck.num_residual_blocks):
                self.resblocks.append(
                    ResidualBlock(
                        ctx.neck.max_num_channels, 
                        ctx.neck.max_num_channels, 
                        norm=ctx.neck.norm
                    )
                )
                
            self.roi_align = torchvision.ops.RoIAlign(
                output_size= ctx.roi_pool.output_size,
                spatial_scale=ctx.roi_pool.spatial_scale,
                sampling_ratio=ctx.roi_pool.sampling_ratio, 
                aligned=ctx.roi_pool.aligned
            )
            
            self.out_proj = nn.Linear(ctx.neck.input_dim, ctx.output_dim)
        elif self.type == 'frame2spike':
            self.neck = SNN_VGG(
                architecture=ctx.neck.architecture,
                img_size=ctx.neck.img_size,
                in_channels=ctx.neck.in_channels,
                out_channels=ctx.neck.out_channels,
                kernel_size=ctx.neck.kernel_size,
                bn=ctx.neck.bn,
                alpha=ctx.neck.alpha,
                timesteps=ctx.neck.timesteps,
                leak_mem=ctx.neck.leak_mem,
                def_threshold=ctx.neck.def_threshold,
                thl=ctx.neck.thl,
                full_prop=ctx.neck.full_prop,
                pretrained=ctx.neck.pretrained
            )
            
            self.roi_align = torchvision.ops.RoIAlign(
                output_size= ctx.roi_pool.output_size,
                spatial_scale=ctx.roi_pool.spatial_scale,
                sampling_ratio=ctx.roi_pool.sampling_ratio, 
                aligned=ctx.roi_pool.aligned
            )
            
            self.out_proj = nn.Linear(ctx.neck.out_channels, ctx.output_dim)

    def forward(self, event_representation, instance_mask_gt, instance_img_gt):
        if self.type == "frame2voxel":
            self.neck.last_states_for_each_channel = {'grayscale': None}
            with torch.no_grad():
                for i in range(20):
                    event_tensor = event_representation[:, i * 5:(i + 1) * 5, :, :]
                    _, _, feat_collate = self.neck.update_reconstruction(event_tensor)
                    
            feat = feat_collate[8]
            for res in self.resblocks:
                feat = res(feat)
        elif self.type == "frame2recon" or self.type == "frame2spike":
            feat = self.neck(event_representation)

        if feat.shape[-1] != self.ctx.semantic_width:
            feat = torch.nn.functional.interpolate(
                feat,
                size=self.ctx.semantic_width,
                mode='bilinear'
            )
            
        loss = {}
        loss['total'] = []
        for i, (feat_per_image, mask_per_image) in enumerate(zip(feat, instance_mask_gt)):
            roi_boxes, _ = self.get_batch_roi_bboxes(mask_per_image)
            
            roi_features = self.roi_align(
                feat_per_image[None, :, :, :],
                roi_boxes,
            ).mean(dim=-1).mean(dim=-1)
            roi_features = self.out_proj(roi_features)
            
            mask_feat_norm = torch.nn.functional.normalize(roi_features, p=2.0, dim=-1)
            
            loss['total'].append(
                loss_func(mask_feat_norm, instance_img_gt[i], self.ctx.loss)
            )
        
        loss['total'] = torch.sum(torch.stack(loss['total']))
        return loss
    
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
    
    def predict(
        self, 
        event_representation=None,
        input_points=None, 
        input_labels=None, 
        input_box=None, 
        classifier=None
    ):
        
        masks, scores = self.predict_mask(
            input_points=input_points, 
            input_labels=input_labels,
            input_box=input_box
        )

        if len(masks) == 0:
            return None, None, None, -1
            

        labels, class_scores = self.predict_label(
            event_representation,
            masks, 
            classifier
        )
            
        total_scores = scores * class_scores
        idx = total_scores.argmax()

        return masks, labels, total_scores, idx
    
    def predict_mask(self, input_points=None, input_labels=None, input_box=None):
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_box,
            multimask_output=True,
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
        
        return masks, scores
    
    def predict_label(self, event_representation, masks, classifier=None):
        if self.type == "feature_crop":
            mask_feature = []
            for mask in masks:
                mask_feature.append(event_representation[mask, :].mean(dim=0))
            mask_feature = torch.stack(mask_feature, dim=0)

            mask_feature_norm = torch.nn.functional.normalize(mask_feature, p=2.0, dim=-1)
            costmap = (mask_feature_norm @ classifier.t() * 100).softmax(dim=-1)
            scores, labels = costmap.max(dim=-1)
        elif self.type == "image_crop":
            mask_feature_norm = []
            for mask in masks:
                images_crop = []
                for lvl in range(3):
                    x1, y1, x2, y2 = self.mask2box_multi_level(mask, lvl, 0.1)
                    cropped_img = event_representation.crop((x1, y1, x2, y2))
                    images_crop.append(self.clip_preprocess(cropped_img))
                image_input = torch.tensor(np.stack(images_crop))
                
                with torch.no_grad():
                    image_features = self.clip.encode_image(image_input.to(self.device)).mean(dim=0, keepdim=True)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
            
                mask_feature_norm.append(image_features)
            mask_feature_norm = torch.cat(mask_feature_norm, dim=0).float()
            costmap = (mask_feature_norm @ classifier.t() * 100).softmax(dim=-1)
        
            scores, labels = costmap.max(dim=-1)
        elif self.type == "label_crop":
            costmap_lst = []
            for mask in masks:
                costmap = event_representation[mask].mean(dim=0)
                costmap_lst.append(costmap)
            costmap = torch.stack(costmap_lst, dim=0)    
            scores, labels = costmap.max(dim=-1)
        elif "frame" in self.type:
            if self.type == "frame2voxel":
                event_representation = event_representation[None, :, :, :]
                self.neck.last_states_for_each_channel = {'grayscale': None}
                
                for i in range(20):
                    event_tensor = event_representation[:, i * 5:(i + 1) * 5, :, :]
                    _, _, feat_collate = self.neck.update_reconstruction(event_tensor)
                    
                feat = feat_collate[8]
                for res in self.resblocks:
                    feat = res(feat)
                    
                # feat = self.proj(feat)
            elif self.type == "frame2recon" or self.type == "frame2spike":
                event_representation = event_representation[None, :, :, :]
                feat = self.neck(event_representation)

            if feat.shape[-1] != self.ctx.semantic_width:
                feat = torch.nn.functional.interpolate(
                    feat,
                    size=self.ctx.semantic_width,
                    mode='bilinear'
                )

            roi_boxes, _ = self.get_batch_roi_bboxes(masks)
            roi_features = self.roi_align(
                feat,
                roi_boxes,
            ).mean(dim=-1).mean(dim=-1)
            roi_features = self.out_proj(roi_features)
            
            mask_feature_norm = torch.nn.functional.normalize(roi_features, p=2.0, dim=-1)
            costmap = (mask_feature_norm @ classifier.t() * 100).softmax(dim=-1)
            scores, labels = costmap.max(dim=-1)
        return labels, scores
    
    def mask2box(self, mask: torch.Tensor):
        row = torch.nonzero(mask.sum(axis=0))[:, 0]
        if len(row) == 0:
            return None
        x1 = row.min().item()
        x2 = row.max().item()
        col = np.nonzero(mask.sum(axis=1))[:, 0]
        y1 = col.min().item()
        y2 = col.max().item()
        return x1, y1, x2 + 1, y2 + 1

    def mask2box_multi_level(self, mask: torch.Tensor, level, expansion_ratio):
        x1, y1, x2 , y2  = self.mask2box(mask)
        if level == 0:
            return x1, y1, x2, y2
        shape = mask.shape
        x_exp = int(abs(x2- x1)*expansion_ratio) * level
        y_exp = int(abs(y2-y1)*expansion_ratio) * level
        return max(0, x1 - x_exp), max(0, y1 - y_exp), min(shape[1], x2 + x_exp), min(shape[0], y2 + y_exp)

class FeatureAdapter(nn.Module):
    def __init__(
            self, 
            c_in, 
            reduction=4,
            ratio=0.2
        ):
        super(FeatureAdapter, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )
        self.ratio = ratio

    def forward(self, x):
        x_refined = self.fc(x)
        return self.ratio * x_refined + (1 - self.ratio) * x
    