from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from .model import (
    apply_special_transform,
    deltas_to_params,
    project_deltas,
    GeometryAwareHierarchicalEncoder,
)
from .segmentation_model import compute_segmentation_metrics
from .training_data import Z_SCALE

VOLUME_FEATURE_DIM = 14


class SharedAttentionSegmentationHead(nn.Module):
    """Shared attention segmentation head (SASH).

    Shared three-view segmentation using the 3x3 own-view / directed-neighbor
    feature matrix. Attribute names preserve the released checkpoint keys.
    """

    def __init__(
        self,
        base_input_dim: int,
        feature_dim: int = 256,
        hidden_dim: int = 256,
        num_context_tokens: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        compatible_heads = min(max(1, int(num_heads)), int(hidden_dim))
        while int(hidden_dim) % compatible_heads != 0:
            compatible_heads -= 1
        self.common_mlp = nn.Sequential(
            nn.Linear(feature_dim * 9, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.row_mlp = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.base_matrix_proj = nn.Sequential(
            nn.Linear(base_input_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.original_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.original_pool = AvgMaxTokenPool(num_tokens=num_context_tokens)
        self.common_pool = AvgMaxTokenPool(num_tokens=num_context_tokens)
        self.view_cross_attention = nn.MultiheadAttention(
            hidden_dim, compatible_heads, dropout=dropout, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.fusion_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        base_features: torch.Tensor,
        local_features: torch.Tensor,
        directed_neighbor_features: torch.Tensor,
    ) -> torch.Tensor:
        if local_features.shape[1] != 3 or directed_neighbor_features.shape[1:3] != (
            3,
            2,
        ):
            raise ValueError(
                "NineFeatureCrossAttentionSegmentationHead expects local features shaped (B,3,N,C) and directed neighbors shaped (B,3,2,N,C)"
            )
        local_originals = [local_features[:, view_idx, :, :] for view_idx in range(3)]
        base_originals = [
            self.base_matrix_proj(base_features[:, view_idx, :, :])
            for view_idx in range(3)
        ]
        nine_features = []
        for view_idx in range(3):
            nine_features.extend(
                [
                    base_originals[view_idx],
                    directed_neighbor_features[:, view_idx, 0, :, :],
                    directed_neighbor_features[:, view_idx, 1, :, :],
                ]
            )
        common_features = self.common_mlp(torch.cat(nine_features, dim=-1))
        common_tokens = self.common_pool(common_features)
        logits = []
        for view_idx in range(3):
            row_features = self.row_mlp(
                torch.cat(
                    [
                        base_originals[view_idx],
                        directed_neighbor_features[:, view_idx, 0, :, :],
                        directed_neighbor_features[:, view_idx, 1, :, :],
                    ],
                    dim=-1,
                )
            )
            original_tokens = self.original_pool(
                self.original_proj(local_originals[view_idx])
            )
            attention_query = common_features
            attention_update, _ = self.view_cross_attention(
                attention_query, original_tokens, common_tokens, need_weights=False
            )
            fused = self.attention_norm(row_features + attention_update)
            fused = self.fusion_norm(fused + self.fusion_ffn(fused))
            logits.append(self.classifier(fused))
        return torch.stack(logits, dim=1)


class AvgMaxTokenPool(nn.Module):
    """Average/max adaptive pooling used by the released model."""

    def __init__(self, num_tokens: int) -> None:
        super().__init__()
        self.num_tokens = int(num_tokens)

    def forward(self, local_features: torch.Tensor) -> torch.Tensor:
        x = local_features.transpose(1, 2).contiguous()
        pooled_avg = F.adaptive_avg_pool1d(x, self.num_tokens)
        pooled_max = F.adaptive_max_pool1d(x, self.num_tokens)
        pooled = 0.5 * (pooled_avg + pooled_max)
        return pooled.transpose(1, 2).contiguous()


class ParameterCouplingAttentionHead(nn.Module):
    """Six parameter embeddings and self-attention produce the registration correction."""

    def __init__(
        self,
        hidden_dim: int,
        num_params: int = 6,
        num_heads: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_params = int(num_params)
        self.param_seed = nn.Linear(hidden_dim, hidden_dim * self.num_params)
        self.param_embedding = nn.Parameter(torch.zeros(self.num_params, hidden_dim))
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.param_embedding, mean=0.0, std=0.02)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        batch = h.shape[0]
        tokens = self.param_seed(h).view(batch, self.num_params, -1)
        tokens = tokens + self.param_embedding.unsqueeze(0)
        update, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm1(tokens + update)
        tokens = self.norm2(tokens + self.ffn(tokens))
        return self.out(tokens).squeeze(-1)


class PairwiseConditionedMoERegistrationHead(nn.Module):
    """Pair-level FiLM modulation, four routed experts, parameter heads and attention correction. Not the unused parameter-level Dynamic Modulation head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_experts: int = 4,
    ) -> None:
        super().__init__()
        if int(input_dim) % 6 != 0:
            raise ValueError(
                "PairwiseConditionedMoERegistrationHead expects six equally sized input feature groups."
            )
        pair_dim = int(input_dim) // 6
        self.num_experts = int(num_experts)
        self.base_proj = nn.Sequential(
            nn.Linear(pair_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.condition_proj = nn.Sequential(
            nn.Linear(pair_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                for _ in range(self.num_experts)
            ]
        )
        self.router = nn.Sequential(
            nn.Linear(pair_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_experts),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.scale_head = nn.Linear(hidden_dim, 2)
        self.shear_head = nn.Linear(hidden_dim, 1)
        self.translation_head = nn.Linear(hidden_dim, 3)
        self.coupling_head = ParameterCouplingAttentionHead(
            hidden_dim=hidden_dim, num_params=6, num_heads=2, dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (
            source_global,
            target_global,
            source_context,
            target_context,
            context_delta,
            context_product,
        ) = x.chunk(6, dim=-1)
        base = torch.cat(
            [source_global, target_global, source_context, target_context], dim=-1
        )
        condition = torch.cat([context_delta, context_product], dim=-1)
        h = self.base_proj(base)
        gamma, beta = self.condition_proj(condition).chunk(2, dim=-1)
        h = h * (1.0 + torch.tanh(gamma)) + beta
        expert_updates = torch.stack([expert(h) for expert in self.experts], dim=1)
        expert_weights = torch.softmax(self.router(condition), dim=-1).unsqueeze(-1)
        h = self.out_norm(h + torch.sum(expert_updates * expert_weights, dim=1))
        scale = self.scale_head(h)
        primary = torch.cat(
            [
                scale[:, 0:1],
                self.shear_head(h),
                scale[:, 1:2],
                self.translation_head(h),
            ],
            dim=-1,
        )
        raw = primary + self.coupling_head(h)
        deltas = project_deltas(raw)
        return deltas_to_params(deltas)


class DynamicModulationRegistrationHead(nn.Module):
    """Dynamic Modulation Registration Head (DMRH).

    Shared relation-conditioned MoE implementation used by the released weight.
    Keep attribute names unchanged to preserve checkpoint parameter keys.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_experts: int = 4,
        num_relations: int = 4,
    ) -> None:
        super().__init__()
        self.relation_embedding = nn.Embedding(int(num_relations), int(input_dim))
        self.base_head = PairwiseConditionedMoERegistrationHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_experts=num_experts,
        )
        nn.init.normal_(self.relation_embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, relation_ids: torch.Tensor) -> torch.Tensor:
        relation_bias = self.relation_embedding(
            relation_ids.to(device=x.device, dtype=torch.long)
        )
        return self.base_head(x + relation_bias.to(dtype=x.dtype))


class SelfAttentionBlock(nn.Module):
    """Transformer block applied to the three-view token sequence within each state."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        update, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm1(tokens + update)
        tokens = self.norm2(tokens + self.ffn(tokens))
        return tokens


class MVMNet(nn.Module):
    """Fixed epoch-98 architecture. Inputs include volume conditions; this is not a point-cloud-only model. Module attribute names preserve checkpoint keys."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_dim = int(256)
        self.normal_k = int(8)
        self.use_normals = bool(False)
        self.use_volume_token = bool(True)
        self.use_registered_seg_aggregation = bool(True)
        self.registered_seg_aggregation_k = int(3)
        self.registered_seg_aggregation_sigma = float(0.08)
        self.registered_seg_aggregation_chunk_size = int(1024)
        self.normal_channels = 0
        self.token_pool_type = str("legacy").strip().lower()
        self.encoder = GeometryAwareHierarchicalEncoder(
            input_dim=3 + self.normal_channels,
            feature_dim=256,
            knn_k=8,
            multi_scale_k=(8, 16),
        )
        self.token_pool = AvgMaxTokenPool(num_tokens=256)
        self.state_blocks = nn.ModuleList(
            [SelfAttentionBlock(dim=256, num_heads=4, dropout=0.1) for _ in range(2)]
        )
        self.cross_state_blocks = nn.ModuleList([])
        self.role_embedding = nn.Embedding(3, 256)
        self.state_embedding = nn.Embedding(2, 256)
        self.reg_relation_names = (
            "empty_aux1_to_target",
            "empty_aux2_to_target",
            "full_aux1_to_target",
            "full_aux2_to_target",
        )
        self.use_shared_reg_head = str(
            "shared_pairwise_conditioned_moe"
        ).strip().lower() in {
            "shared_pairwise_conditioned_moe",
            "shared-pairwise-conditioned-moe",
            "shared_pairwise_moe",
        }
        self.volume_token_mlp = nn.Sequential(
            nn.Linear(VOLUME_FEATURE_DIM, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
        )
        self.registered_seg_context_mlp = nn.Sequential(
            nn.Linear(256 + 1 + 6, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
        )
        self.shared_reg_head = DynamicModulationRegistrationHead(
            input_dim=256 * 6,
            hidden_dim=256,
            dropout=0.1,
            num_relations=len(self.reg_relation_names),
        )
        self.reg_heads = nn.ModuleDict()
        seg_input_dim = 3 + self.normal_channels + 256 * 4 + 2
        self.seg_head = SharedAttentionSegmentationHead(
            base_input_dim=seg_input_dim,
            feature_dim=256,
            hidden_dim=256,
            num_context_tokens=256,
            num_heads=4,
            dropout=0.1,
        )
        self.refiner = None

    def _aggregate_registered_neighbors(
        self,
        query_points: torch.Tensor,
        key_points: torch.Tensor,
        key_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, query_count, _ = query_points.shape
        channels = key_features.shape[-1]
        k = min(
            max(1, int(self.registered_seg_aggregation_k)), int(key_points.shape[1])
        )
        chunk_size = max(1, int(self.registered_seg_aggregation_chunk_size))
        sigma2 = max(float(self.registered_seg_aggregation_sigma) ** 2, 1e-06)
        contexts = []
        distances = []
        for start in range(0, query_count, chunk_size):
            query_chunk = query_points[:, start : start + chunk_size, :]
            dists = torch.cdist(query_chunk, key_points)
            top_dist, top_idx = torch.topk(dists, k=k, dim=-1, largest=False)
            weights = torch.softmax(-0.5 * top_dist.square() / sigma2, dim=-1)
            expanded_features = key_features.unsqueeze(1).expand(
                -1, query_chunk.shape[1], -1, -1
            )
            neighbor_features = torch.gather(
                expanded_features,
                dim=2,
                index=top_idx.unsqueeze(-1).expand(-1, -1, -1, channels),
            )
            contexts.append(torch.sum(neighbor_features * weights.unsqueeze(-1), dim=2))
            distances.append(torch.sum(top_dist * weights, dim=-1, keepdim=True))
        return (torch.cat(contexts, dim=1), torch.cat(distances, dim=1))

    def _project_registered_seg_context(
        self,
        local_features: torch.Tensor,
        neighbor_features: torch.Tensor,
        neighbor_distance: torch.Tensor,
        reg_params: torch.Tensor,
    ) -> torch.Tensor:
        if self.registered_seg_context_mlp is None:
            return torch.zeros_like(local_features)
        batch, num_points, _ = local_features.shape
        reg_token = reg_params.unsqueeze(1).expand(-1, num_points, -1)
        features = torch.cat([neighbor_features, neighbor_distance, reg_token], dim=-1)
        return self.registered_seg_context_mlp(
            features.reshape(batch * num_points, -1)
        ).view(batch, num_points, -1)

    def _build_registered_seg_contexts(
        self,
        registered_points: torch.Tensor,
        local_stack: torch.Tensor,
        view_reg_params: list[torch.Tensor],
    ) -> torch.Tensor:
        batch, num_views, num_points, channels = local_stack.shape
        contexts = local_stack.new_zeros((batch, num_views, 2, num_points, channels))
        if self.registered_seg_context_mlp is None or local_stack.shape[1] < 3:
            return contexts
        for query_idx in range(3):
            neighbor_slot = 0
            for key_idx in range(3):
                if query_idx == key_idx:
                    continue
                (
                    neighbor_features,
                    neighbor_distance,
                ) = self._aggregate_registered_neighbors(
                    query_points=registered_points[:, query_idx, :, :],
                    key_points=registered_points[:, key_idx, :, :],
                    key_features=local_stack[:, key_idx, :, :],
                )
                relative_reg_params = params_to_deltas(
                    view_reg_params[key_idx]
                ) - params_to_deltas(view_reg_params[query_idx])
                contexts[
                    :, query_idx, neighbor_slot, :, :
                ] = self._project_registered_seg_context(
                    local_features=local_stack[:, query_idx, :, :],
                    neighbor_features=neighbor_features,
                    neighbor_distance=neighbor_distance,
                    reg_params=relative_reg_params,
                )
                neighbor_slot += 1
        return contexts

    def _build_volume_features(
        self,
        volume_m3: torch.Tensor | None,
        volume_quality: torch.Tensor | None,
        volume_geometry: torch.Tensor | None,
        volume_mask: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.volume_token_mlp is None:
            return None
        volume = torch.zeros((batch_size, 1), dtype=dtype, device=device)
        quality = torch.zeros((batch_size, 3), dtype=dtype, device=device)
        geometry = torch.zeros((batch_size, 10), dtype=dtype, device=device)
        mask = torch.ones((batch_size, 1), dtype=dtype, device=device)
        if volume_m3 is not None:
            volume = volume_m3.to(device=device, dtype=dtype).view(batch_size, -1)[
                :, :1
            ]
        if volume_quality is not None:
            quality = volume_quality.to(device=device, dtype=dtype).view(
                batch_size, -1
            )[:, :3]
        if volume_geometry is not None:
            geometry = volume_geometry.to(device=device, dtype=dtype).view(
                batch_size, -1
            )[:, :10]
        if volume_mask is not None:
            mask = volume_mask.to(device=device, dtype=dtype).view(batch_size, -1)[
                :, :1
            ]
        normalized_quality = quality.clone()
        normalized_quality[:, 2] = normalized_quality[:, 2] / 1000.0
        normalized_geometry = geometry.clone()
        normalized_geometry[:, [1, 3]] = normalized_geometry[:, [1, 3]] / float(Z_SCALE)
        normalized_geometry[:, 4:7] = normalized_geometry[:, 4:7] / 1000.0
        normalized_geometry[:, 7] = normalized_geometry[:, 7] / 10.0
        normalized_geometry[:, 8:10] = normalized_geometry[:, 8:10] / 100000.0
        features = torch.cat(
            [
                torch.log1p(volume.clamp_min(0.0)),
                normalized_quality,
                normalized_geometry,
            ],
            dim=-1,
        )
        return features * mask.clamp(0.0, 1.0)

    def _encode_state_tokens(
        self,
        state_points: torch.Tensor,
        state_index: int,
        volume_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, num_views, _, _ = state_points.shape
        globals_per_view = []
        locals_per_view = []
        normals_per_view = []
        pooled_tokens = []
        state_ids = torch.full(
            (batch,), int(state_index), dtype=torch.long, device=state_points.device
        )
        state_embed = self.state_embedding(state_ids).unsqueeze(1)
        for view_index in range(num_views):
            points = state_points[:, view_index, :, :]
            encoder_input = points[:, :, :3]
            normals = points.new_empty((*points.shape[:2], 0))
            global_feat, local_feat = self.encoder.encode(encoder_input)
            tokens = self.token_pool(local_feat)
            role_ids = torch.full(
                (batch,), int(view_index), dtype=torch.long, device=state_points.device
            )
            role_embed = self.role_embedding(role_ids).unsqueeze(1)
            tokens = tokens + role_embed + state_embed
            globals_per_view.append(global_feat)
            locals_per_view.append(local_feat)
            normals_per_view.append(normals)
            pooled_tokens.append(tokens)
        global_stack = torch.stack(globals_per_view, dim=1)
        local_stack = torch.stack(locals_per_view, dim=1)
        normal_stack = torch.stack(normals_per_view, dim=1)
        token_stack = torch.stack(pooled_tokens, dim=1)
        batch, num_views, num_tokens, channels = token_stack.shape
        flat_tokens = token_stack.view(batch, num_views * num_tokens, channels)
        if self.volume_token_mlp is not None and volume_features is not None:
            volume_token = (
                self.volume_token_mlp(volume_features).unsqueeze(1) + state_embed
            )
            flat_tokens = torch.cat([flat_tokens, volume_token], dim=1)
        for block in self.state_blocks:
            flat_tokens = block(flat_tokens)
        flat_tokens = flat_tokens[:, : num_views * num_tokens, :]
        token_stack = flat_tokens.view(batch, num_views, num_tokens, channels)
        return (global_stack, local_stack, normal_stack, token_stack)

    def _summarize_tokens(
        self, token_stack: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        view_context = token_stack.mean(dim=2)
        state_summary = view_context.mean(dim=1)
        return (view_context, state_summary)

    def _encode_state(
        self,
        state_points: torch.Tensor,
        state_index: int,
        volume_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            global_stack,
            local_stack,
            normal_stack,
            token_stack,
        ) = self._encode_state_tokens(
            state_points, state_index=state_index, volume_features=volume_features
        )
        view_context, state_summary = self._summarize_tokens(token_stack)
        return (global_stack, local_stack, normal_stack, view_context, state_summary)

    def _build_reg_input(
        self,
        source_global: torch.Tensor,
        target_global: torch.Tensor,
        source_context: torch.Tensor,
        target_context: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                source_global,
                target_global,
                source_context,
                target_context,
                source_context - target_context,
                source_context * target_context,
            ],
            dim=-1,
        )

    def _build_seg_features(
        self,
        points: torch.Tensor,
        normals: torch.Tensor,
        local_features: torch.Tensor,
        view_global: torch.Tensor,
        state_summary: torch.Tensor,
        cross_context: torch.Tensor,
        state_value: float,
        role_value: float,
    ) -> torch.Tensor:
        batch, num_points, _ = points.shape
        view_global_expanded = view_global.unsqueeze(1).expand(-1, num_points, -1)
        state_summary_expanded = state_summary.unsqueeze(1).expand(-1, num_points, -1)
        cross_context_expanded = cross_context.unsqueeze(1).expand(-1, num_points, -1)
        state_token = torch.full(
            (batch, num_points, 1),
            float(state_value),
            dtype=points.dtype,
            device=points.device,
        )
        role_token = torch.full(
            (batch, num_points, 1),
            float(role_value),
            dtype=points.dtype,
            device=points.device,
        )
        features = [
            points,
            normals,
            local_features,
            view_global_expanded,
            state_summary_expanded,
            cross_context_expanded,
        ]
        features.extend([state_token, role_token])
        return torch.cat(features, dim=-1)

    def _build_multiview_seg_logits(
        self,
        original_points: torch.Tensor,
        registered_points: torch.Tensor,
        local_stack: torch.Tensor,
        normal_stack: torch.Tensor,
        global_stack: torch.Tensor,
        view_context: torch.Tensor,
        state_summary: torch.Tensor,
        view_reg_params: list[torch.Tensor],
        state_value: float,
    ) -> torch.Tensor:
        directed_neighbor_features = self._build_registered_seg_contexts(
            registered_points=registered_points,
            local_stack=local_stack,
            view_reg_params=view_reg_params,
        )
        base_features = []
        for view_idx in range(3):
            base_features.append(
                self._build_seg_features(
                    points=original_points[:, view_idx, :, :],
                    normals=normal_stack[:, view_idx, :, :],
                    local_features=local_stack[:, view_idx, :, :],
                    view_global=global_stack[:, view_idx, :],
                    state_summary=state_summary,
                    cross_context=view_context[:, view_idx, :],
                    state_value=state_value,
                    role_value=float(view_idx),
                )
            )
        return self.seg_head(
            base_features=torch.stack(base_features, dim=1),
            local_features=local_stack,
            directed_neighbor_features=directed_neighbor_features,
        )

    def forward(
        self,
        empty_points: torch.Tensor,
        full_points: torch.Tensor,
        volume_m3: torch.Tensor | None = None,
        volume_quality: torch.Tensor | None = None,
        volume_geometry: torch.Tensor | None = None,
        volume_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        volume_features = self._build_volume_features(
            volume_m3=volume_m3,
            volume_quality=volume_quality,
            volume_geometry=volume_geometry,
            volume_mask=volume_mask,
            batch_size=int(empty_points.shape[0]),
            device=empty_points.device,
            dtype=empty_points.dtype,
        )
        (
            empty_global,
            empty_local,
            empty_normals,
            empty_context,
            empty_summary,
        ) = self._encode_state(
            empty_points, state_index=0, volume_features=volume_features
        )
        (
            full_global,
            full_local,
            full_normals,
            full_context,
            full_summary,
        ) = self._encode_state(
            full_points, state_index=1, volume_features=volume_features
        )
        reg_specs = [
            ("empty_aux1_to_target", "empty", 1, 0),
            ("empty_aux2_to_target", "empty", 2, 0),
            ("full_aux1_to_target", "full", 1, 0),
            ("full_aux2_to_target", "full", 2, 0),
        ]
        reg_params_list = []
        registered_empty = empty_points.clone()
        registered_full = full_points.clone()
        identity_reg_params = empty_points.new_zeros((empty_points.shape[0], 6))
        identity_reg_params[:, 0] = 1.0
        identity_reg_params[:, 2] = 1.0
        empty_view_reg_params = [
            identity_reg_params,
            identity_reg_params,
            identity_reg_params,
        ]
        full_view_reg_params = [
            identity_reg_params,
            identity_reg_params,
            identity_reg_params,
        ]
        for relation_idx, (head_name, state_name, source_idx, target_idx) in enumerate(
            reg_specs
        ):
            if state_name == "empty":
                global_stack = empty_global
                context_stack = empty_context
                points_stack = empty_points
            else:
                global_stack = full_global
                context_stack = full_context
                points_stack = full_points
            reg_input = self._build_reg_input(
                source_global=global_stack[:, source_idx, :],
                target_global=global_stack[:, target_idx, :],
                source_context=context_stack[:, source_idx, :],
                target_context=context_stack[:, target_idx, :],
            )
            relation_ids = torch.full(
                (reg_input.shape[0],),
                int(relation_idx),
                dtype=torch.long,
                device=reg_input.device,
            )
            params = self.shared_reg_head(reg_input, relation_ids)
            transformed = apply_special_transform(
                points_stack[:, source_idx, :, :], params
            )
            reg_params_list.append(params)
            if state_name == "empty":
                registered_empty[:, source_idx, :, :] = transformed
                empty_view_reg_params[source_idx] = params
            else:
                registered_full[:, source_idx, :, :] = transformed
                full_view_reg_params[source_idx] = params
        empty_seg_logits = self._build_multiview_seg_logits(
            original_points=empty_points,
            registered_points=registered_empty,
            local_stack=empty_local,
            normal_stack=empty_normals,
            global_stack=empty_global,
            view_context=empty_context,
            state_summary=empty_summary,
            view_reg_params=empty_view_reg_params,
            state_value=0.0,
        )
        full_seg_logits = self._build_multiview_seg_logits(
            original_points=full_points,
            registered_points=registered_full,
            local_stack=full_local,
            normal_stack=full_normals,
            global_stack=full_global,
            view_context=full_context,
            state_summary=full_summary,
            view_reg_params=full_view_reg_params,
            state_value=1.0,
        )
        outputs = {
            "reg_params": torch.stack(reg_params_list, dim=1),
            "registered_empty_points": registered_empty,
            "registered_full_points": registered_full,
            "empty_seg_logits": empty_seg_logits,
            "full_seg_logits": full_seg_logits,
        }
        return outputs


def params_to_deltas(params: torch.Tensor) -> torch.Tensor:
    deltas = params.clone()
    deltas[:, 0] = params[:, 0] - 1.0
    deltas[:, 1] = params[:, 1]
    deltas[:, 2] = params[:, 2] - 1.0
    deltas[:, 3:] = params[:, 3:]
    return deltas


def trimmed_chamfer_distance(
    x: torch.Tensor, y: torch.Tensor, trim_ratio: float = 0.9
) -> torch.Tensor:
    dists = torch.cdist(x, y)
    d_xy = dists.min(dim=2).values
    d_yx = dists.min(dim=1).values
    keep_x = max(1, int(d_xy.shape[1] * trim_ratio))
    keep_y = max(1, int(d_yx.shape[1] * trim_ratio))
    d_xy_trim = torch.topk(d_xy, k=keep_x, dim=1, largest=False).values.mean(dim=1)
    d_yx_trim = torch.topk(d_yx, k=keep_y, dim=1, largest=False).values.mean(dim=1)
    return d_xy_trim + d_yx_trim


def _compute_masked_segmentation_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    view_mask: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = logits.device
    valid = view_mask > 0.5
    if not bool(torch.any(valid)):
        zero = logits.new_tensor(0.0)
        return (
            zero,
            {
                "accuracy": 0.0,
                "mean_iou": 0.0,
                "box_iou": 0.0,
                "surface_iou": 0.0,
                "valid_views": 0.0,
            },
        )
    valid_logits = logits[valid]
    valid_labels = labels[valid]
    loss = F.cross_entropy(
        valid_logits.reshape(-1, valid_logits.shape[-1]),
        valid_labels.reshape(-1),
        weight=class_weights,
    )
    metrics = compute_segmentation_metrics(valid_logits.detach(), valid_labels)
    metrics["valid_views"] = float(valid.sum().item())
    return (loss.to(device), metrics)


def _denormalize_state_points(
    points: torch.Tensor, target_center: torch.Tensor, target_scale: torch.Tensor
) -> torch.Tensor:
    scale = target_scale.view(target_scale.shape[0], 1, 1, 1)
    center = target_center.view(target_center.shape[0], 1, 1, 3)
    return points * scale + center


def _select_volume_points(
    points: torch.Tensor, weights: torch.Tensor, max_points: int
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, count, _ = points.shape
    limit = int(max_points)
    if limit <= 0 or count <= limit:
        return (points, weights)
    idx = torch.linspace(0, count - 1, steps=limit, device=points.device).round().long()
    return (points.index_select(1, idx), weights.index_select(1, idx))


def _soft_surface_volume_m3(
    empty_points: torch.Tensor,
    full_points: torch.Tensor,
    empty_weights: torch.Tensor,
    full_weights: torch.Tensor,
    domain_area_mm2: torch.Tensor,
    sigma_mm: float,
    max_points: int,
) -> torch.Tensor:
    empty_points, empty_weights = _select_volume_points(
        empty_points, empty_weights, max_points=max_points
    )
    full_points, full_weights = _select_volume_points(
        full_points, full_weights, max_points=max_points
    )
    eps = torch.finfo(empty_points.dtype).eps
    empty_yz = empty_points[:, :, 1:3]
    full_yz = full_points[:, :, 1:3]
    d2 = torch.cdist(empty_yz, full_yz).square()
    sigma2 = max(float(sigma_mm) ** 2, 1e-06)
    logits = -0.5 * d2 / sigma2
    logits = logits + torch.log(full_weights.clamp_min(eps)).unsqueeze(1)
    interp_weights = torch.softmax(logits, dim=-1)
    full_x = torch.sum(interp_weights * full_points[:, :, 0].unsqueeze(1), dim=-1)
    full_support = torch.sum(interp_weights * full_weights.unsqueeze(1), dim=-1)
    x_distance = torch.sqrt((full_x - empty_points[:, :, 0]).square() + 0.0001)
    usable_weight = empty_weights * full_support
    mean_distance = torch.sum(usable_weight * x_distance, dim=1) / usable_weight.sum(
        dim=1
    ).clamp_min(eps)
    volume_mm3 = mean_distance * domain_area_mm2.to(
        dtype=empty_points.dtype, device=empty_points.device
    ).view(-1)
    return volume_mm3 / 1000000000.0


def _apply_volume_label_geometry(
    points: torch.Tensor, geometry: torch.Tensor, is_full: bool
) -> torch.Tensor:
    """Map model-space points into the same z/y-aligned space used for volume labels."""
    if geometry.ndim == 1:
        geometry = geometry.unsqueeze(0)
    z_scale_idx = 2 if is_full else 0
    z_offset_idx = 3 if is_full else 1
    z_scale = (
        geometry[:, z_scale_idx]
        .to(dtype=points.dtype, device=points.device)
        .view(-1, 1, 1)
    )
    z_offset = (
        geometry[:, z_offset_idx]
        .to(dtype=points.dtype, device=points.device)
        .view(-1, 1, 1)
    )
    mapped = points.clone()
    mapped[:, :, 2:3] = mapped[:, :, 2:3] / float(Z_SCALE) * z_scale + z_offset
    if is_full:
        shift = (
            geometry[:, 4:7].to(dtype=points.dtype, device=points.device).view(-1, 1, 3)
        )
        mapped = mapped + shift
    return mapped


def _compute_multiview_volume_loss(
    batch: dict[str, torch.Tensor | str],
    outputs: dict[str, torch.Tensor],
    volume_scale_m3: float,
    volume_sigma_mm: float,
    volume_max_points: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    target = batch.get("volume_m3")
    mask = batch.get("volume_mask")
    geometry = batch.get("volume_geometry")
    if (
        not isinstance(target, torch.Tensor)
        or not isinstance(mask, torch.Tensor)
        or (not isinstance(geometry, torch.Tensor))
    ):
        zero = outputs["empty_seg_logits"].new_tensor(0.0)
        return (
            zero,
            {
                "volume_valid": 0.0,
                "volume_pred_m3": 0.0,
                "volume_target_m3": 0.0,
                "volume_rel_mae": 0.0,
            },
        )
    valid = mask.view(-1) > 0.5
    if not bool(torch.any(valid)):
        zero = outputs["empty_seg_logits"].new_tensor(0.0)
        return (
            zero,
            {
                "volume_valid": 0.0,
                "volume_pred_m3": 0.0,
                "volume_target_m3": 0.0,
                "volume_rel_mae": 0.0,
            },
        )
    empty_world = _denormalize_state_points(
        outputs["registered_empty_points"],
        batch["empty_target_center"],
        batch["empty_target_scale"],
    )
    full_world = _denormalize_state_points(
        outputs["registered_full_points"],
        batch["full_target_center"],
        batch["full_target_scale"],
    )
    geometry = geometry.to(dtype=empty_world.dtype, device=empty_world.device)
    empty_volume_space = _apply_volume_label_geometry(
        empty_world.flatten(1, 2), geometry, is_full=False
    )
    full_volume_space = _apply_volume_label_geometry(
        full_world.flatten(1, 2), geometry, is_full=True
    )
    empty_surface_prob = torch.softmax(outputs["empty_seg_logits"], dim=-1)[..., 0]
    full_surface_prob = torch.softmax(outputs["full_seg_logits"], dim=-1)[..., 0]
    empty_valid = (
        batch["empty_valid_mask"].unsqueeze(-1).to(dtype=empty_surface_prob.dtype)
    )
    full_valid = (
        batch["full_valid_mask"].unsqueeze(-1).to(dtype=full_surface_prob.dtype)
    )
    empty_weights = (empty_surface_prob * empty_valid).flatten(1)
    full_weights = (full_surface_prob * full_valid).flatten(1)
    grid_step = geometry[:, 7].clamp_min(1.0)
    usable_grid_samples = geometry[:, 9].clamp_min(1.0)
    domain_area_mm2 = usable_grid_samples * grid_step.square()
    pred_volume = _soft_surface_volume_m3(
        empty_points=empty_volume_space,
        full_points=full_volume_space,
        empty_weights=empty_weights,
        full_weights=full_weights,
        domain_area_mm2=domain_area_mm2,
        sigma_mm=float(volume_sigma_mm),
        max_points=int(volume_max_points),
    )
    target_volume = target.view(-1).to(
        dtype=pred_volume.dtype, device=pred_volume.device
    )
    denom = (
        pred_volume.new_tensor(float(volume_scale_m3))
        if float(volume_scale_m3) > 0.0
        else target_volume.detach().abs().clamp_min(0.001)
    )
    rel_error = (pred_volume - target_volume) / denom
    loss = F.smooth_l1_loss(rel_error[valid], torch.zeros_like(rel_error[valid]))
    rel_mae = torch.mean(
        torch.abs(
            (pred_volume[valid] - target_volume[valid])
            / target_volume[valid].detach().abs().clamp_min(0.001)
        )
    )
    return (
        loss,
        {
            "volume_valid": float(valid.sum().detach().cpu()),
            "volume_pred_m3": float(pred_volume[valid].mean().detach().cpu()),
            "volume_target_m3": float(target_volume[valid].mean().detach().cpu()),
            "volume_rel_mae": float(rel_mae.detach().cpu()),
        },
    )


def _nearest_log_probs(
    query_points: torch.Tensor,
    key_points: torch.Tensor,
    key_log_probs: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    batch, query_count, _ = query_points.shape
    channels = key_log_probs.shape[-1]
    chunks = []
    chunk = max(1, int(chunk_size))
    for start in range(0, query_count, chunk):
        query_chunk = query_points[:, start : start + chunk, :]
        nn_idx = torch.cdist(query_chunk, key_points).argmin(dim=-1)
        expanded = key_log_probs.unsqueeze(1).expand(-1, query_chunk.shape[1], -1, -1)
        chunks.append(
            torch.gather(
                expanded,
                dim=2,
                index=nn_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, channels),
            ).squeeze(2)
        )
    return torch.cat(chunks, dim=1)


def _compute_registered_seg_consistency_loss(
    state_points: torch.Tensor,
    registered_points: torch.Tensor,
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    reg_mask: torch.Tensor,
    head_offset: int,
    chunk_size: int,
) -> tuple[torch.Tensor, float]:
    losses = []
    target_points = state_points[:, 0, :, :]
    target_log_probs = torch.log_softmax(logits[:, 0, :, :], dim=-1)
    for source_idx, head_idx in [(1, head_offset), (2, head_offset + 1)]:
        sample_valid = (
            (valid_mask[:, 0] > 0.5)
            & (valid_mask[:, source_idx] > 0.5)
            & (reg_mask[:, head_idx] > 0.5)
        )
        if not bool(torch.any(sample_valid)):
            continue
        source_log_probs = torch.log_softmax(logits[:, source_idx, :, :], dim=-1)
        source_probs = source_log_probs.exp()
        nearest_target_log_probs = _nearest_log_probs(
            query_points=registered_points[:, source_idx, :, :],
            key_points=target_points,
            key_log_probs=target_log_probs,
            chunk_size=chunk_size,
        )
        point_kl = torch.sum(
            source_probs * (source_log_probs - nearest_target_log_probs), dim=-1
        )
        losses.append(point_kl[sample_valid].mean())
    if not losses:
        return (logits.new_tensor(0.0), 0.0)
    return (torch.stack(losses).mean(), float(len(losses)))


def compute_multiview_fusion_losses(
    batch: dict[str, torch.Tensor | str],
    outputs: dict[str, torch.Tensor],
    param_weights: torch.Tensor,
    trim_ratio: float = 0.9,
    reg_geom_weight: float = 2.0,
    reg_param_weight: float = 1.0,
    reg_transform_weight: float = 2.0,
    seg_weight: float = 1.0,
    seg_class_weights: torch.Tensor | None = None,
    volume_loss_weight: float = 0.0,
    volume_loss_scale_m3: float = 0.0,
    volume_loss_sigma_mm: float = 80.0,
    volume_loss_max_points: int = 1024,
    seg_consistency_weight: float = 0.0,
    seg_consistency_chunk_size: int = 1024,
    boundary_loss_weight: float = 0.0,
    boundary_knn_k: int = 16,
    seg_guided_reg_weight: float = 0.0,
    seg_guided_reg_surface_weight: float = 1.0,
    seg_guided_reg_box_weight: float = 1.0,
    divider_reg_weight: float = 0.0,
    divider_side_trim_frac: float = 0.08,
    divider_end_trim_frac: float = 0.06,
    seg_guided_reg_max_points: int = 2048,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_reg_params = outputs["reg_params"]
    target_reg_params = batch["reg_head_params"]
    reg_mask = batch["reg_head_mask"]
    reg_param_loss = pred_reg_params.new_tensor(0.0)
    reg_transform_loss = pred_reg_params.new_tensor(0.0)
    reg_geom_loss = pred_reg_params.new_tensor(0.0)
    seg_guided_reg_loss = pred_reg_params.new_tensor(0.0)
    seg_guided_reg_surface_loss = 0.0
    seg_guided_reg_box_loss = 0.0
    divider_reg_loss = pred_reg_params.new_tensor(0.0)
    divider_reg_valid = 0.0
    valid_reg = reg_mask > 0.5
    if bool(torch.any(valid_reg)):
        pred_flat = pred_reg_params[valid_reg]
        target_flat = target_reg_params[valid_reg]
        pred_deltas = params_to_deltas(pred_flat)
        target_deltas = params_to_deltas(target_flat)
        weighted_diff = F.smooth_l1_loss(pred_deltas, target_deltas, reduction="none")
        reg_param_loss = (weighted_diff * param_weights.unsqueeze(0)).mean()
        target_empty = batch["empty_points"][:, 0, :, :]
        target_full = batch["full_points"][:, 0, :, :]
        pred_registered = torch.stack(
            [
                outputs["registered_empty_points"][:, 1, :, :],
                outputs["registered_empty_points"][:, 2, :, :],
                outputs["registered_full_points"][:, 1, :, :],
                outputs["registered_full_points"][:, 2, :, :],
            ],
            dim=1,
        )
        target_registered = torch.stack(
            [target_empty, target_empty, target_full, target_full], dim=1
        )
        pred_registered_valid = pred_registered[valid_reg]
        target_registered_valid = target_registered[valid_reg]
        reg_transform_loss = F.smooth_l1_loss(
            pred_registered_valid, target_registered_valid
        )
        reg_geom_loss = trimmed_chamfer_distance(
            pred_registered_valid, target_registered_valid, trim_ratio=trim_ratio
        ).mean()
    empty_seg_loss, empty_seg_metrics = _compute_masked_segmentation_loss(
        logits=outputs["empty_seg_logits"],
        labels=batch["empty_seg_labels"],
        view_mask=batch["empty_seg_mask"],
        class_weights=seg_class_weights,
    )
    full_seg_loss, full_seg_metrics = _compute_masked_segmentation_loss(
        logits=outputs["full_seg_logits"],
        labels=batch["full_seg_labels"],
        view_mask=batch["full_seg_mask"],
        class_weights=seg_class_weights,
    )
    valid_seg_sets = max(
        int(empty_seg_metrics["valid_views"] > 0.0)
        + int(full_seg_metrics["valid_views"] > 0.0),
        1,
    )
    seg_loss = (empty_seg_loss + full_seg_loss) / float(valid_seg_sets)
    boundary_loss = pred_reg_params.new_tensor(0.0)
    boundary_metrics = {
        "boundary_loss": 0.0,
        "boundary_accuracy": 0.0,
        "boundary_precision": 0.0,
        "boundary_recall": 0.0,
        "boundary_iou": 0.0,
        "boundary_seg_ce": 0.0,
        "boundary_positive_rate": 0.0,
        "boundary_pred_positive_rate": 0.0,
    }
    volume_loss = pred_reg_params.new_tensor(0.0)
    volume_metrics = {
        "volume_valid": 0.0,
        "volume_pred_m3": 0.0,
        "volume_target_m3": 0.0,
        "volume_rel_mae": 0.0,
    }
    if float(volume_loss_weight) > 0.0:
        volume_loss, volume_metrics = _compute_multiview_volume_loss(
            batch=batch,
            outputs=outputs,
            volume_scale_m3=float(volume_loss_scale_m3),
            volume_sigma_mm=float(volume_loss_sigma_mm),
            volume_max_points=int(volume_loss_max_points),
        )
    seg_consistency_loss = pred_reg_params.new_tensor(0.0)
    seg_consistency_valid = 0.0
    if float(seg_consistency_weight) > 0.0:
        (
            empty_consistency,
            empty_consistency_valid,
        ) = _compute_registered_seg_consistency_loss(
            state_points=batch["empty_points"],
            registered_points=outputs["registered_empty_points"],
            logits=outputs["empty_seg_logits"],
            valid_mask=batch["empty_valid_mask"],
            reg_mask=reg_mask,
            head_offset=0,
            chunk_size=int(seg_consistency_chunk_size),
        )
        (
            full_consistency,
            full_consistency_valid,
        ) = _compute_registered_seg_consistency_loss(
            state_points=batch["full_points"],
            registered_points=outputs["registered_full_points"],
            logits=outputs["full_seg_logits"],
            valid_mask=batch["full_valid_mask"],
            reg_mask=reg_mask,
            head_offset=2,
            chunk_size=int(seg_consistency_chunk_size),
        )
        valid_sets = max(
            int(empty_consistency_valid > 0.0) + int(full_consistency_valid > 0.0), 1
        )
        seg_consistency_loss = (empty_consistency + full_consistency) / float(
            valid_sets
        )
        seg_consistency_valid = empty_consistency_valid + full_consistency_valid
    total = (
        reg_param_weight * reg_param_loss
        + reg_transform_weight * reg_transform_loss
        + reg_geom_weight * reg_geom_loss
        + float(seg_guided_reg_weight) * seg_guided_reg_loss
        + float(divider_reg_weight) * divider_reg_loss
        + seg_weight * seg_loss
        + float(boundary_loss_weight) * boundary_loss
        + float(volume_loss_weight) * volume_loss
        + float(seg_consistency_weight) * seg_consistency_loss
    )
    if empty_seg_metrics["valid_views"] > 0.0 and full_seg_metrics["valid_views"] > 0.0:
        seg_accuracy = 0.5 * (
            empty_seg_metrics["accuracy"] + full_seg_metrics["accuracy"]
        )
        seg_mean_iou = 0.5 * (
            empty_seg_metrics["mean_iou"] + full_seg_metrics["mean_iou"]
        )
        seg_box_iou = 0.5 * (empty_seg_metrics["box_iou"] + full_seg_metrics["box_iou"])
        seg_surface_iou = 0.5 * (
            empty_seg_metrics["surface_iou"] + full_seg_metrics["surface_iou"]
        )
    elif empty_seg_metrics["valid_views"] > 0.0:
        seg_accuracy = empty_seg_metrics["accuracy"]
        seg_mean_iou = empty_seg_metrics["mean_iou"]
        seg_box_iou = empty_seg_metrics["box_iou"]
        seg_surface_iou = empty_seg_metrics["surface_iou"]
    elif full_seg_metrics["valid_views"] > 0.0:
        seg_accuracy = full_seg_metrics["accuracy"]
        seg_mean_iou = full_seg_metrics["mean_iou"]
        seg_box_iou = full_seg_metrics["box_iou"]
        seg_surface_iou = full_seg_metrics["surface_iou"]
    else:
        seg_accuracy = 0.0
        seg_mean_iou = 0.0
        seg_box_iou = 0.0
        seg_surface_iou = 0.0
    metrics = {
        "loss": float(total.detach().cpu()),
        "reg_param_loss": float(reg_param_loss.detach().cpu()),
        "reg_transform_loss": float(reg_transform_loss.detach().cpu()),
        "reg_geom_loss": float(reg_geom_loss.detach().cpu()),
        "seg_guided_reg_loss": float(seg_guided_reg_loss.detach().cpu()),
        "seg_guided_reg_surface_loss": float(seg_guided_reg_surface_loss),
        "seg_guided_reg_box_loss": float(seg_guided_reg_box_loss),
        "divider_reg_loss": float(divider_reg_loss.detach().cpu()),
        "divider_reg_valid": float(divider_reg_valid),
        "seg_loss": float(seg_loss.detach().cpu()),
        **boundary_metrics,
        "seg_consistency_loss": float(seg_consistency_loss.detach().cpu()),
        "seg_consistency_valid": float(seg_consistency_valid),
        "volume_loss": float(volume_loss.detach().cpu()),
        "seg_accuracy": float(seg_accuracy),
        "seg_mean_iou": float(seg_mean_iou),
        "seg_box_iou": float(seg_box_iou),
        "seg_surface_iou": float(seg_surface_iou),
        **volume_metrics,
    }
    return (total, metrics)
