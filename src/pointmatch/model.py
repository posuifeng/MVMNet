from __future__ import annotations
import torch
from torch import nn


class ConvGNReLU1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups != 0 and groups > 1:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualPointBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(8, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        self.net = nn.Sequential(
            ConvGNReLU1d(channels, channels),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.net(x))


def _knn_indices(points: torch.Tensor, k: int) -> torch.Tensor:
    num_points = int(points.shape[1])
    if num_points <= 1:
        return torch.zeros(
            (*points.shape[:2], 1), dtype=torch.long, device=points.device
        )
    k_eff = max(1, min(int(k), num_points - 1))
    dists = torch.cdist(points, points)
    return torch.topk(dists, k=k_eff + 1, dim=-1, largest=False).indices[:, :, 1:]


def _gather_neighbors(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    batch, num_points, channels = features.shape
    k = idx.shape[-1]
    batch_offsets = (
        torch.arange(batch, device=features.device).view(batch, 1, 1) * num_points
    )
    flat_idx = (idx + batch_offsets).reshape(-1)
    flat_features = features.reshape(batch * num_points, channels)
    return flat_features[flat_idx].view(batch, num_points, k, channels)


class EdgeConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups != 0 and groups > 1:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        neighbors = _gather_neighbors(features, idx)
        center = features.unsqueeze(2).expand_as(neighbors)
        edge = torch.cat([center, neighbors - center], dim=-1)
        edge = edge.permute(0, 3, 1, 2).contiguous()
        return self.net(edge).max(dim=-1).values


class GeometryAwareHierarchicalEncoder(nn.Module):
    """Local/global point encoder of GAHE: multi-scale EdgeConv (8,16), residual fusion, and max/mean/attention pooling. Role/view tokens and two Transformer blocks are implemented in MVMNet."""

    def __init__(
        self,
        input_dim: int = 3,
        feature_dim: int = 256,
        knn_k: int = 16,
        multi_scale_k: tuple[int, ...] = (8, 16, 32),
    ) -> None:
        super().__init__()
        self.knn_k = int(knn_k)
        self.multi_scale_k = tuple((int(v) for v in multi_scale_k if int(v) > 0))
        if not self.multi_scale_k:
            self.multi_scale_k = (self.knn_k,)
        first_channels = max(32, feature_dim // 4)
        mid_channels = max(64, feature_dim // 2)
        self.scale_blocks = nn.ModuleList(
            [EdgeConvBlock(input_dim, first_channels) for _ in self.multi_scale_k]
        )
        self.scale_project = nn.Sequential(
            ConvGNReLU1d(first_channels * len(self.multi_scale_k), mid_channels),
            ResidualPointBlock(mid_channels),
        )
        self.edge2 = EdgeConvBlock(mid_channels, mid_channels)
        self.edge3 = EdgeConvBlock(mid_channels, feature_dim)
        self.fuse = nn.Sequential(
            ConvGNReLU1d(mid_channels + mid_channels + feature_dim, feature_dim),
            ResidualPointBlock(feature_dim),
        )
        self.attn = nn.Conv1d(feature_dim, 1, kernel_size=1)
        self.project = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
        )

    def encode(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xyz = points[:, :, :3]
        max_k = max((*self.multi_scale_k, self.knn_k))
        idx_max = _knn_indices(xyz, max_k)
        scale_features = []
        for k, block in zip(self.multi_scale_k, self.scale_blocks):
            idx = idx_max[:, :, : max(1, min(int(k), idx_max.shape[-1]))]
            scale_features.append(block(points, idx))
        x1 = (
            self.scale_project(torch.cat(scale_features, dim=1))
            .transpose(1, 2)
            .contiguous()
        )
        idx = idx_max[:, :, : max(1, min(self.knn_k, idx_max.shape[-1]))]
        x2 = self.edge2(x1, idx).transpose(1, 2).contiguous()
        x3 = self.edge3(x2, idx).transpose(1, 2).contiguous()
        features = self.fuse(
            torch.cat([x1, x2, x3], dim=-1).transpose(1, 2).contiguous()
        )
        max_pool = torch.max(features, dim=2).values
        mean_pool = torch.mean(features, dim=2)
        attn = torch.softmax(self.attn(features), dim=2)
        attn_pool = torch.sum(features * attn, dim=2)
        global_feature = self.project(
            torch.cat([max_pool, mean_pool, attn_pool], dim=1)
        )
        local_features = features.transpose(1, 2).contiguous()
        return (global_feature, local_features)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        global_feature, _ = self.encode(points)
        return global_feature


def project_deltas(raw: torch.Tensor) -> torch.Tensor:
    out = raw.clone()
    out[:, 0] = 0.2 * torch.tanh(raw[:, 0])
    out[:, 1] = 0.35 * torch.tanh(raw[:, 1])
    out[:, 2] = 0.2 * torch.tanh(raw[:, 2])
    out[:, 3:] = raw[:, 3:]
    return out


def deltas_to_params(deltas: torch.Tensor) -> torch.Tensor:
    params = deltas.clone()
    params[:, 0] = 1.0 + deltas[:, 0]
    params[:, 1] = deltas[:, 1]
    params[:, 2] = 1.0 + deltas[:, 2]
    params[:, 3:] = deltas[:, 3:]
    return params


def build_matrix_from_params(params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch = params.shape[0]
    matrix = torch.zeros((batch, 3, 3), dtype=params.dtype, device=params.device)
    matrix[:, 0, 0] = params[:, 0]
    matrix[:, 1, 1] = 1.0
    matrix[:, 2, 0] = params[:, 1]
    matrix[:, 2, 2] = params[:, 2]
    translation = params[:, 3:]
    return (matrix, translation)


def apply_special_transform(points: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    matrix, translation = build_matrix_from_params(params)
    return torch.bmm(points, matrix.transpose(1, 2)) + translation.unsqueeze(1)
