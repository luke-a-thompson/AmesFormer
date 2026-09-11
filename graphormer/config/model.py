from dataclasses import dataclass
from collections import OrderedDict
from typing import Dict, List

import torch.nn as nn
from graphormer.modules.attention import (
    AttentionPrior,
    AttentionPadMask,
    GraphormerMultiHeadAttention,
)
from graphormer.modules.embedding import EdgeEmbedding, NodeEmbedding
from graphormer.modules.encoding import SpatialEncoding, EdgeEncoding
from graphormer.modules.layers import FeedForwardNetwork, Output
from graphormer.modules.model import Graphormer
from graphormer.modules.encoding import CentralityEncoding
from graphormer.config.options import AttentionType, NormType, ResidualType
from graphormer.modules.norm.identity import IdentityNorm
from graphormer.modules.norm import LayerNorm, RMSNorm, CRMSNorm, MaxNorm
from graphormer.modules.residuals import PreNormResidual, RezeroResidual


@dataclass
class ModelConfig:
    num_layers: int | None = None
    node_feature_dim: int | None = None
    hidden_dim: int | None = None
    edge_feature_dim: int | None = None
    edge_embedding_dim: int | None = None
    ffn_hidden_dim: int | None = None
    output_dim: int | None = None
    n_heads: int | None = None
    heads_by_layer: List[int] | None = None
    max_in_degree: int | None = None
    max_out_degree: int | None = None
    max_path_distance: int | None = None
    dropout: float | None = None
    state_dict: Dict | None = None
    norm_type: NormType | None = None
    attention_type: AttentionType | None = None
    residual_type: ResidualType | None = None

    def build(self, *, node_feature_dim=None, edge_feature_dim=None, output_dim=None) -> Graphormer:
        for field, value in (("node_feature_dim", node_feature_dim), ("edge_feature_dim", edge_feature_dim), ("output_dim", output_dim)):
            if value is not None:
                setattr(self, field, value)
        for field in ("num_layers", "node_feature_dim", "hidden_dim", "edge_feature_dim",
                      "edge_embedding_dim", "ffn_hidden_dim", "output_dim", "max_in_degree",
                      "max_out_degree", "max_path_distance", "dropout", "norm_type", "residual_type"):
            if getattr(self, field) is None:
                raise AttributeError(f"{field} is not defined for Graphormer")
        if self.attention_type not in (None, AttentionType.MHA):
            raise ValueError("Only MHA attention is supported")
        heads = self.heads_by_layer or [self.n_heads] * self.num_layers
        if len(heads) != self.num_layers or any(h is None or h < 1 for h in heads):
            raise ValueError("Provide one positive head count per layer, or n_heads")
        norm_class = {NormType.LAYER: LayerNorm, NormType.RMS: RMSNorm, NormType.CRMS: CRMSNorm,
                      NormType.MAX: MaxNorm, NormType.NONE: lambda _: IdentityNorm()}[self.norm_type]
        residual_layers = OrderedDict()
        for i, num_heads in enumerate(heads):
            input_norm = norm_class(self.hidden_dim)
            attention = GraphormerMultiHeadAttention(num_heads, self.hidden_dim, self.dropout)
            attention_norm = norm_class(self.hidden_dim)
            ffn = FeedForwardNetwork(self.hidden_dim, self.ffn_hidden_dim, self.hidden_dim, self.dropout)
            if self.residual_type == ResidualType.PRENORM:
                residual = PreNormResidual(input_norm, attention, attention_norm, ffn)
            elif self.residual_type == ResidualType.REZERO:
                residual = RezeroResidual(attention, ffn)
            else:
                raise ValueError(f"Unknown residual type: {self.residual_type}")
            residual_layers[f"residual_layer_{i}"] = residual

        modules = OrderedDict()

        modules["node_embedding"] = NodeEmbedding(self.node_feature_dim, self.hidden_dim)
        modules["edge_embedding"] = EdgeEmbedding(self.edge_feature_dim, self.edge_embedding_dim)
        modules["centrality_encoding"] = CentralityEncoding(self.max_in_degree, self.max_out_degree, self.hidden_dim)
        modules["spatial_encoding"] = SpatialEncoding(self.max_path_distance)
        modules["edge_encoding"] = EdgeEncoding(self.edge_embedding_dim, self.max_path_distance)
        modules["attention_prior"] = AttentionPrior()
        modules["attention_pad_mask"] = AttentionPadMask()
        for key, residual_layer in residual_layers.items():
            modules[key] = residual_layer
        modules["output"] = Output(self.hidden_dim, self.output_dim)

        model = Graphormer(modules)

        if self.state_dict is not None:
            model.load_state_dict(self.state_dict)

        return model
