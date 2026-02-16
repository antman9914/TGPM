import copy
import time
import os.path as osp
import numpy as np
import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F
# from utils.eval import *
from utils.utils import get_device_from_model


class PatternEncoder(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim):
        super(PatternEncoder, self).__init__()
        self.input_dim = input_dim
        self.edge_dim = edge_dim
        self.hidden_dim = (hidden_dim - 100) // 2
        self.num_heads = 1
        self.num_layers = 1

        self.raw_input_dim = self.input_dim + self.edge_dim

        self.pre_projection = nn.Linear(self.raw_input_dim, self.hidden_dim, bias=False)
        self.node_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self._init_encoder()

    def _init_encoder(self):
        # Initialize pattern encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.projection = nn.Identity() 


    def _encode_features(self, feat_gathered, mask=None):
        if mask is not None:
            mask = ~mask
        else:
            mask = None
        embeddings = self.encoder(feat_gathered, src_key_padding_mask=mask)
        return embeddings, 0.


    def encode_node(self, dataset, patterns, feat, e_feat=None):

        device = get_device_from_model(self)
        h, b, n, k = patterns.shape

        patterns_flat = patterns.view(-1)  # Shape: [h * n * k]
        feat_gathered = feat[patterns_flat].to(device)  # Shape: [h * n * k, d]
        feat_gathered = feat_gathered.view(h * n * b, k, -1)  # Reshape to [h * n, k, d]

        src_feat = feat_gathered[:, 0, :]
        feat_gathered = feat_gathered[:, 1:, :]

        if e_feat is not None:
            ed = e_feat.shape[-1]
            e_feat_gathered = e_feat.view(h * n * b, k - 1, ed)

            feat_gathered = torch.cat([feat_gathered, e_feat_gathered], dim=-1)
        else:
            feat_gathered = torch.cat([feat_gathered, torch.zeros(h * n * b, k, self.edge_dim, device=device)], dim=-1)

        feat_gathered = self.pre_projection(feat_gathered)
        src_feat = self.node_proj(src_feat)

        pattern_feat, _ = self._encode_features(feat_gathered, None)

        pattern_feat = pattern_feat.mean(dim=1).view(h, b * n, self.hidden_dim)
        src_feat = src_feat.view(h, b*n, self.hidden_dim)
        pattern_feat = torch.cat([pattern_feat, src_feat], dim=-1)
        
        return pattern_feat
        