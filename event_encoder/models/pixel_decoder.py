# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

from models.layers import Conv2d
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F


def get_decoder(ctx):
    if ctx.name == "UpsamplerDecoder":
        return UpsamplerDecoder(
            in_channel=ctx.in_channel,
            embed_dim=ctx.embed_dim,
            out_channel=ctx.out_channel,
            align_corners=ctx.align_corners,
            multilayers=ctx.multilayers
        )
    elif ctx.name == "UpsamplerDecoder128":
        return UpsamplerDecoder128(
            in_channel=ctx.in_channel,
            embed_dim=ctx.embed_dim,
            out_channel=ctx.out_channel,
            align_corners=ctx.align_corners,
            multilayers=ctx.multilayers
        )
    elif ctx.name == "UpsamplerDecoderx2":
        return UpsamplerDecoderx2(
            in_channel=ctx.in_channel,
            embed_dim=ctx.embed_dim,
            out_channel=ctx.out_channel,
            align_corners=ctx.align_corners,
            multilayers=ctx.multilayers
        )


def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.

    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "GN": lambda channels: nn.GroupNorm(32, channels),
        }[norm]
    return norm(out_channels)


class BasePixelDecoder(nn.Module):
    def __init__(
        self,
        feature_channels,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()

        lateral_convs = nn.ModuleList()
        output_convs = nn.ModuleList()

        use_bias = norm == ""
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(feature_channels) - 1:
                output_norm = get_norm(norm, conv_dim)
                output_conv = Conv2d(
                    in_channels,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(None)
                output_convs.append(output_conv)
            else:
                lateral_norm = get_norm(norm, conv_dim)
                output_norm = get_norm(norm, conv_dim)

                lateral_conv = Conv2d(
                    in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
                )
                output_conv = Conv2d(
                    conv_dim,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                weight_init.c2_xavier_fill(lateral_conv)
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("adapter_{}".format(idx + 1), lateral_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(lateral_conv)
                output_convs.append(output_conv)

        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        self.mask_dim = mask_dim
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        weight_init.c2_xavier_fill(self.mask_features)

    def forward(self, features):
        for idx, feature in enumerate(features[::-1]):
            x = feature
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            if lateral_conv is None:
                y = output_conv(x)
            else:
                cur_fpn = lateral_conv(x)
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)
        return self.mask_features(y)


class UpsamplerDecoder128(nn.Module):
    def __init__(
        self,
        in_channel: int = 768,
        embed_dim: int = 256,
        out_channel: int = 768,
        align_corners: bool = False,
        multilayers: bool = False
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()

        self.conv_0 = nn.Conv2d(in_channel, embed_dim, kernel_size=3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(embed_dim, out_channel, kernel_size=3, stride=1, padding=1)

        self.syncbn_fc_0 = nn.SyncBatchNorm(embed_dim)
        self.syncbn_fc_1 = nn.SyncBatchNorm(embed_dim)

        self.align_corners = align_corners
        
        self.multilayers = multilayers
        if self.multilayers:
            self.proj_0 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False)
            self.proj_1 = nn.Conv2d(embed_dim, out_channel, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        x_collate = []
        
        x_collate.append(x)
        
        x = self.conv_0(x)
        x = self.syncbn_fc_0(x)
        x = F.relu(x, inplace=True)
        x = F.interpolate(x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)
        
        x_collate.append(x)
        
        x = self.conv_1(x)
        x = self.syncbn_fc_1(x)
        x = F.relu(x, inplace=True)
        x = self.conv_2(x)
        x = F.interpolate(x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)

        x_collate.append(x)
    
        if self.multilayers:
            proj_collate = [self.proj_0, self.proj_1]   
            return [proj(x) for proj, x in zip(proj_collate, x_collate[:-1])] + [x_collate[-1]]
        else:
            return x_collate[-1]


class UpsamplerDecoder(nn.Module):
    def __init__(
        self,
        in_channel: int = 768,
        embed_dim: int = 256,
        out_channel: int = 768,
        align_corners: bool = False,
        multilayers: bool = False
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()

        self.conv_0 = nn.Conv2d(in_channel, embed_dim, kernel_size=3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(embed_dim, out_channel, kernel_size=1, stride=1)

        self.syncbn_fc_0 = nn.SyncBatchNorm(embed_dim)
        self.syncbn_fc_1 = nn.SyncBatchNorm(embed_dim)
        self.syncbn_fc_2 = nn.SyncBatchNorm(embed_dim)

        self.align_corners = align_corners
        
        self.multilayers = multilayers
        if self.multilayers:
            self.proj_0 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False)
            self.proj_1 = nn.Conv2d(embed_dim, out_channel, kernel_size=1, stride=1, bias=False)
            self.proj_2 = nn.Conv2d(embed_dim, out_channel, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        x_collate = []
        
        x_collate.append(x)
        
        x = self.conv_0(x)
        x = self.syncbn_fc_0(x)
        x = F.relu(x, inplace=True)
        x = F.interpolate(x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)
        
        x_collate.append(x)
        
        x = self.conv_1(x)
        x = self.syncbn_fc_1(x)
        x = F.relu(x, inplace=True)
        x = F.interpolate(x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)
        
        x_collate.append(x)
        
        x = self.conv_2(x)
        x = self.syncbn_fc_2(x)
        x = F.relu(x, inplace=True)
        x = self.conv_3(x)
        x = F.interpolate(x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)
        
        x_collate.append(x)
        
        if self.multilayers:
            proj_collate = [self.proj_0, self.proj_1, self.proj_2]   
            return [proj(x) for proj, x in zip(proj_collate, x_collate[:-1])] + [x_collate[-1]]
        else:
            return x_collate[-1]


class UpsamplerDecoderx2(nn.Module):
    def __init__(
        self,
        in_channel: int = 768,
        embed_dim: int = 256,
        out_channel: int = 768,
        align_corners: bool = False,
        multilayers: bool = False
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()

        self.conv_0 = nn.Conv2d(in_channel, embed_dim, kernel_size=3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        self.conv_4 = nn.Conv2d(embed_dim, out_channel, kernel_size=1, stride=1)

        self.syncbn_fc_0 = nn.SyncBatchNorm(embed_dim)
        self.syncbn_fc_1 = nn.SyncBatchNorm(embed_dim)
        self.syncbn_fc_2 = nn.SyncBatchNorm(embed_dim)
        self.syncbn_fc_3 = nn.SyncBatchNorm(embed_dim)

        self.align_corners = align_corners
        
        self.multilayers = multilayers
        if self.multilayers:
            self.proj_0 = nn.Conv2d(embed_dim, out_channel, kernel_size=1, stride=1, bias=False)
            self.proj_1 = nn.Conv2d(embed_dim, out_channel, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        x_collate = []
        
        x = self.conv_0(x)
        x = self.syncbn_fc_0(x)
        x = F.relu(x, inplace=True)
        x = F.interpolate(x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)
        
        x_collate.append(x)
        
        x = self.conv_1(x)
        x = self.syncbn_fc_1(x)
        x = F.relu(x, inplace=True)
        x = F.interpolate(x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)
        
        x_collate.append(x)
        
        x = self.conv_2(x)
        x = self.syncbn_fc_2(x)
        x = F.relu(x, inplace=True)
        x = F.interpolate(x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)
        
        x_collate.append(x)
        
        x = self.conv_3(x)
        x = self.syncbn_fc_3(x)
        x = F.relu(x, inplace=True)
        x = F.interpolate(x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)
        
        x = self.conv_4(x)
        
        x_collate.append(x)
        
        if self.multilayers:
            proj_collate = [self.proj_0, self.proj_1]   
            return [proj(x) for proj, x in zip(proj_collate, x_collate[:-1])] + [x_collate[-1]]
        else:
            return x_collate[-1]
