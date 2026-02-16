import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import get_device_from_model
from .encoder import PatternEncoder
from models.modules import TimeEncoder


def mask_feature(
        x,
        p: float = 0.5,
        mode: str = 'col',
        fill_value: float = 0.,
        training: bool = True,
):
    if p < 0. or p > 1.:
        raise ValueError(f'Masking ratio has to be between 0 and 1 '
                         f'(got {p}')
    if not training or p == 0.0:
        return x, torch.ones_like(x, dtype=torch.bool)
    assert mode in ['row', 'col', 'all']

    if mode == 'row':
        mask = torch.rand(x.size(0), device=x.device) >= p
        mask = mask.view(-1, 1)
    elif mode == 'col':
        mask = torch.rand(x.size(1), device=x.device) >= p
        mask = mask.view(1, -1)
    else:
        mask = torch.rand_like(x) >= p

    x = x.masked_fill(~mask, fill_value)
    return x, mask


def mask_patterns(
        patterns: torch.Tensor,
        p: float = 0.5,
        mode: str = 'mask',  # 'mask' or 'random'
        training: bool = True,
):
    """Mask patterns tensor by either zeroing or randomizing node indices.
    
    Args:
        patterns: Tensor of shape [h, n, k] containing node indices
        p: Probability of masking each position
        mode: 'mask' to mask with -1s, 'random' to replace with random node indices
        training: Whether in training mode
    
    Returns:
        Tuple of (masked patterns, mask boolean tensor)
    """
    if p < 0. or p > 1.:
        raise ValueError(f'Masking ratio has to be between 0 and 1 (got {p})')

    if not training or p == 0.0:
        return patterns, torch.ones_like(patterns, dtype=torch.bool)

    # Create mask of shape [h, n, k]
    mask = torch.rand_like(patterns.float()) >= p

    if mode == 'mask':
        # Mask positions with zeros
        patterns = patterns.masked_fill(~mask, -1)
    elif mode == 'random':
        # Generate random node indices between 0 and n-1
        n = patterns.size(1)
        random_indices = torch.randint_like(patterns, 0, n)
        # Replace masked positions with random indices
        patterns = torch.where(mask, patterns, random_indices)
    else:
        raise ValueError(f"Mode must be 'zero' or 'random', got {mode}")

    return patterns, mask


class BaseModel(nn.Module):
    def __init__(self, node_dim, edge_dim, neighbor_sampler, time_feat_dim, hidden_dim, num_enc_layer, num_heads, num_tokens, dropout):
        super(BaseModel, self).__init__()

        self.node_raw_features = None
        self.edge_raw_features = None
        self.raw_time = None

        self.max_input_sequence_length = num_tokens
        self.neighbor_sampler = neighbor_sampler
        self.time_feat_dim = time_feat_dim
        self.dropout = dropout

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_enc_layers = num_enc_layer 
        self.vq_encoder = PatternEncoder(self.node_dim, self.edge_dim + self.time_feat_dim, self.hidden_dim)
        self.time_encoder = TimeEncoder(time_dim=self.time_feat_dim, parameter_requires_grad=True)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            norm_first=False,
            batch_first=True,
        )
        self.encoder = nn.ModuleList([copy.deepcopy(self.encoder_layer) for _ in range(self.num_enc_layers)])
        self.encoder_norm = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.num_enc_layers)])

        self.register_buffer('pre_transformation', None)

    def linear_probe(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = True

    def set_dataset(self, node_raw_features, edge_raw_features, raw_time):
        self.node_raw_features = torch.from_numpy(node_raw_features).float()
        self.edge_raw_features = torch.from_numpy(edge_raw_features).float()
        np.insert(raw_time, 0, 0)
        self.raw_time = torch.from_numpy(raw_time).float()
        
    def set_pre_transformation(self, input_dim, output_dim):
        self.pre_transformation = nn.Linear(input_dim, output_dim)

    def reset_head(self, output_dim):
        self.head = nn.Linear(self.hidden_dim, output_dim)

    def transformer_encode(self, x, mask=None):
        for layer, norm in zip(self.encoder, self.encoder_norm):
            last_x = x
            x = layer(norm(x))
            x = last_x + x
        return x

    def get_instance_emb(self, pattern_emb, params):
        if params['use_cls_token']:
            instance_emb = pattern_emb[0].squeeze(0)
        else:
            instance_emb = pattern_emb.mean(dim=0)
        return instance_emb

    def set_neighbor_sampler(self, neighbor_sampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()

    def pad_sequences(self, node_ids: np.ndarray, node_interact_times: np.ndarray, nodes_neighbor_ids_list: list, nodes_edge_ids_list: list,
                      nodes_neighbor_times_list: list, patch_size: int = 1):
        """
        pad the sequences for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param nodes_neighbor_ids_list: list of ndarrays, each ndarray contains neighbor ids for nodes in node_ids
        :param nodes_edge_ids_list: list of ndarrays, each ndarray contains edge ids for nodes in node_ids
        :param nodes_neighbor_times_list: list of ndarrays, each ndarray contains neighbor interaction timestamp for nodes in node_ids
        :param patch_size: int, patch size
        :param max_input_sequence_length: int, maximal number of neighbors for each node
        :return:
        """
        assert self.max_input_sequence_length - 1 > 0, 'Maximal number of neighbors for each node should be greater than 1!'
        # first cut the sequence of nodes whose number of neighbors is more than max_input_sequence_length - 1 (we need to include the target node in the sequence)
        for idx in range(len(nodes_neighbor_ids_list)):
            assert len(nodes_neighbor_ids_list[idx]) == len(nodes_edge_ids_list[idx]) == len(nodes_neighbor_times_list[idx])
            if len(nodes_neighbor_ids_list[idx]) > self.max_input_sequence_length - 1:
                # cut the sequence by taking the most recent max_input_sequence_length interactions
                nodes_neighbor_ids_list[idx] = nodes_neighbor_ids_list[idx][-(self.max_input_sequence_length - 1):]
                nodes_edge_ids_list[idx] = nodes_edge_ids_list[idx][-(self.max_input_sequence_length - 1):]
                nodes_neighbor_times_list[idx] = nodes_neighbor_times_list[idx][-(self.max_input_sequence_length - 1):]

        # pad the sequences
        # three ndarrays with shape (batch_size, max_seq_length)
        padded_nodes_neighbor_ids = np.zeros((len(node_ids), self.max_input_sequence_length)).astype(np.longlong)
        padded_nodes_edge_ids = np.zeros((len(node_ids), self.max_input_sequence_length)).astype(np.longlong)
        padded_nodes_neighbor_times = np.zeros((len(node_ids), self.max_input_sequence_length)).astype(np.float32)

        for idx in range(len(node_ids)):

            if len(nodes_neighbor_ids_list[idx]) > 0:
                padded_nodes_neighbor_ids[idx, :len(nodes_neighbor_ids_list[idx])] = nodes_neighbor_ids_list[idx]
                padded_nodes_edge_ids[idx, :len(nodes_edge_ids_list[idx])] = nodes_edge_ids_list[idx]
                padded_nodes_neighbor_times[idx, :len(nodes_neighbor_times_list[idx])] = nodes_neighbor_times_list[idx]

        # three ndarrays with shape (batch_size, max_seq_length)
        return torch.from_numpy(padded_nodes_neighbor_ids), torch.from_numpy(padded_nodes_edge_ids), torch.from_numpy(padded_nodes_neighbor_times)


    def encode_node(self, dataset, patterns, eids, src_list, dst_list, time_list):
        device = get_device_from_model(self)
        src_neighbor_list, src_eid_list, src_time_list = self.neighbor_sampler.get_all_first_hop_neighbors(src_list, time_list)
        dst_neighbor_list, dst_eid_list, dst_time_list = self.neighbor_sampler.get_all_first_hop_neighbors(dst_list, time_list)

        src_pad_nlist, src_pad_elist, src_pad_tlist = self.pad_sequences(src_list, time_list, src_neighbor_list, src_eid_list, src_time_list)
        dst_pad_nlist, dst_pad_elist, dst_pad_tlist = self.pad_sequences(dst_list, time_list, dst_neighbor_list, dst_eid_list, dst_time_list) # num_nodes, num_neighbors

        src_selected_patterns = patterns[:, src_pad_elist, :]
        dst_selected_patterns = patterns[:, dst_pad_elist, :]
        h, num_nodes, num_neighbors, k = src_selected_patterns.shape

        src_patterns = src_selected_patterns
        dst_patterns = dst_selected_patterns

        # Batch Data Extraction
        src_selected_eids = eids[:, src_pad_elist, :]   # [h, b, num_neighbor, walk_length]
        src_selected_times = self.raw_time[src_selected_eids]
        dst_selected_eids = eids[:, dst_pad_elist, :]
        dst_selected_times = self.raw_time[dst_selected_eids]

        src_eids = src_selected_eids
        src_times = src_selected_times.to(device)
        src_efeat = self.edge_raw_features[src_eids].to(device)
        src_eids = src_eids.to(device)
        dst_eids = dst_selected_eids
        dst_times = dst_selected_times.to(device)
        dst_efeat = self.edge_raw_features[dst_eids].to(device)
        dst_eids = dst_eids.to(device)

        # Compute Temporal Encodings based on Relative Time
        time_list = torch.from_numpy(time_list).to(device).view(1, -1, 1, 1)
        src_tdelta = src_times[:, :, :, :1] - src_times
        dst_tdelta = dst_times[:, :, :, :1] - dst_times
        src_temb = self.time_encoder(src_tdelta)
        dst_temb = self.time_encoder(dst_tdelta)
        src_temb[src_times == 0] = 0.0
        dst_temb[dst_times == 0] = 0.0

        time_list = time_list.view(-1, 1)
        src_pad_tlist = src_pad_tlist.to(device)
        src_tdelta_2 = (time_list - src_pad_tlist)
        src_temb_2 = self.time_encoder(src_tdelta_2)
        src_temb_2[src_pad_tlist == 0] = 0.0
        dst_pad_tlist = dst_pad_tlist.to(device)
        dst_tdelta_2 = (time_list - dst_pad_tlist)
        dst_temb_2 = self.time_encoder(dst_tdelta_2)
        dst_temb_2[dst_pad_tlist == 0] = 0.0
        temb_2 = torch.cat([src_temb_2, dst_temb_2], dim=0)

        # Concatenate edge features with temporal encodings
        src_efeat = torch.cat([src_efeat, src_temb], dim=-1)
        dst_efeat = torch.cat([dst_efeat, dst_temb], dim=-1)

        selected_patterns = torch.cat([src_patterns, dst_patterns], dim=1)
        efeat = torch.cat([src_efeat, dst_efeat], dim=1)

        # Pattern Encoder + TGPM Encoder
        pattern_feat = self.vq_encoder.encode_node(dataset, selected_patterns, self.node_raw_features, efeat)
        pattern_feat = pattern_feat.view(h, num_nodes*2, num_neighbors, -1).mean(0)
        pattern_feat = torch.cat([pattern_feat, temb_2], dim=-1)

        neighbor_emb = self.transformer_encode(pattern_feat)
        src_neighbor_emb, dst_neighbor_emb = neighbor_emb[:num_nodes, :, :], neighbor_emb[num_nodes:, :, :]
        src_final_emb, dst_final_emb = src_neighbor_emb.mean(1), dst_neighbor_emb.mean(1)

        return src_final_emb, dst_final_emb, src_pad_nlist.detach().cpu().numpy(), dst_pad_nlist.detach().cpu().numpy()


class PretrainModel(BaseModel):
    def __init__(self, node_dim, edge_dim, neighbor_sampler, time_feat_dim, hidden_dim, num_enc_layer, num_heads, num_tokens, dropout, 
                 num_dec_layer=1, aug_data=False, ntp=True, lt=True, st=True, random_mask=False, block_size=6, visible_ratio=0.25):
        super(PretrainModel, self).__init__(node_dim, edge_dim, neighbor_sampler, time_feat_dim, hidden_dim, num_enc_layer, num_heads, num_tokens, dropout)
        
        self.num_dec_layers = num_dec_layer

        # Initialize mask token
        self.aug_data = aug_data
        self.alpha = 0.
        self.beta = 0.
        if ntp:
            self.beta = 1.
        if lt or st:
            self.alpha = 1.
        self.lt = lt
        self.st = st
        self.random_mask = random_mask
        self.block_size = block_size
        self.visible_ratio = visible_ratio
        self.mask_token = nn.Parameter(torch.zeros(1, self.hidden_dim), requires_grad=True)
        nn.init.normal_(self.mask_token, std=0.02)

        # Create online encoder and norms
        self.online_pattern_encoder = copy.deepcopy(self.vq_encoder)
        self.online_encoder = copy.deepcopy(self.encoder)
        self.online_encoder_norm = copy.deepcopy(self.encoder_norm)

        for module in [self.online_pattern_encoder, self.online_encoder, self.online_encoder_norm]:
            for param in module.parameters():
                param.requires_grad = False

        self.decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            norm_first=False
        )
        self.decoder = nn.ModuleList([copy.deepcopy(self.decoder_layer) for _ in range(self.num_dec_layers)])
        self.decoder_norm = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.num_dec_layers)])

        self.decoder_lin = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.decoder_next_time = nn.Sequential(nn.Linear(self.hidden_dim, 4*self.hidden_dim), nn.ReLU(), nn.Linear(4 * self.hidden_dim, self.time_feat_dim))

    def online_transformer_encode(self, x):
        for layer, norm in zip(self.online_encoder, self.online_encoder_norm):
            x = layer(norm(x)) + x
        return x

    def transformer_decode(self, x):
        for layer, norm in zip(self.decoder, self.decoder_norm):
            x = layer(norm(x)) + x
        return x

    def ema_update(self, alpha=0.99):
        for online_param, param in zip(self.online_pattern_encoder.parameters(), self.vq_encoder.parameters()):
            online_param.data.copy_(online_param.data * alpha + param.data * (1 - alpha))

        for online_layer, layer in zip(self.online_encoder, self.encoder):
            for online_param, param in zip(online_layer.parameters(), layer.parameters()):
                online_param.data.copy_(online_param.data * alpha + param.data * (1 - alpha))

        for online_norm, norm in zip(self.online_encoder_norm, self.encoder_norm):
            for online_param, param in zip(online_norm.parameters(), norm.parameters()):
                online_param.data.copy_(online_param.data * alpha + param.data * (1 - alpha))

    def pretrain_node(self, dataset, patterns, eids, src_list, dst_list, time_list):
        device = get_device_from_model(self)

        src_neighbor_list, src_eid_list, src_time_list = self.neighbor_sampler.get_all_first_hop_neighbors(src_list, time_list)

        src_pad_nlist, src_pad_elist, src_pad_tlist = self.pad_sequences(src_list, time_list, src_neighbor_list, src_eid_list, src_time_list)
        
        src_patterns = patterns[:, src_pad_elist, :]
        h, num_nodes, num_neighbors, k = src_patterns.shape
        
        src_selected_eids = eids[:, src_pad_elist, :]   # [h, b, num_neighbor, walk_length]
        src_selected_times = self.raw_time[src_selected_eids]
        src_eids = src_selected_eids
        src_times = src_selected_times.to(device)
        src_efeat = self.edge_raw_features[src_eids].to(device)
        src_eids = src_eids.to(device)

        # Get temporal encoding of each token
        time_list = torch.from_numpy(time_list).to(device).view(1, -1, 1, 1)
        src_tdelta = src_times[:, :, :, :1] - src_times
        src_temb = self.time_encoder(src_tdelta)
        src_temb[src_times == 0] = 0.0

        time_list = time_list.view(-1, 1)
        src_pad_tlist = src_pad_tlist.to(device)
        src_tdelta_2 = time_list - src_pad_tlist
        src_temb_2 = self.time_encoder(src_tdelta_2)
        src_temb_2[src_pad_tlist == 0] = 0.0

        src_efeat = torch.cat([src_efeat, src_temb], dim=-1)

        # Get patterns
        total_pattern_count = num_neighbors
        visible_pattern_count = int(num_neighbors * self.visible_ratio)
        masked_pattern_count = total_pattern_count - visible_pattern_count
        block_num = masked_pattern_count // self.block_size
        total_block_size = total_pattern_count // block_num

        # Code for TGPM block-masking based pre-train
        # Shuffle patterns and use top k patterns
        if not self.random_mask:
            # For default long-term masking, we sample a consecutive block with size masked_pattern_count for masking.
            shuffle_idx = torch.arange(total_pattern_count)
            start_idx = torch.randperm(visible_pattern_count)[0]
            shuffle_idx = torch.cat([shuffle_idx[:start_idx], shuffle_idx[start_idx+masked_pattern_count:], shuffle_idx[start_idx:start_idx+masked_pattern_count]], -1)
            unshuffle_idx = torch.argsort(shuffle_idx)
            mask = torch.zeros(total_pattern_count, dtype=torch.bool, device=device)
            mask[shuffle_idx[visible_pattern_count:]] = True

            # For short-term masking, we randomly sample a set of smaller blocks with specified size for masking.
            ref_idx = torch.arange(total_pattern_count)
            start_idx_2 = torch.randint(0, total_block_size - self.block_size + 1, (block_num,))
            visivle_idx, shuffle_idx_2 = [], []
            for i in range(len(start_idx_2)):
                visivle_idx.append(ref_idx[total_block_size*i:total_block_size*i+start_idx_2[i]])
                if i == block_num - 1:
                    visivle_idx.append(ref_idx[total_block_size*i+start_idx_2[i]+self.block_size:])
                else:
                    visivle_idx.append(ref_idx[total_block_size*i+start_idx_2[i]+self.block_size:total_block_size*(i+1)])
                shuffle_idx_2.append(ref_idx[total_block_size*i+start_idx_2[i]:total_block_size*i+start_idx_2[i]+self.block_size])
            visivle_idx = torch.cat(visivle_idx, -1)
            shuffle_idx_2 = torch.cat(shuffle_idx_2, -1)
            shuffle_idx_2 = torch.cat([visivle_idx, shuffle_idx_2], -1)
            unshuffle_idx_2 = torch.argsort(shuffle_idx_2)
            mask_2 = torch.zeros(total_pattern_count, dtype=torch.bool, device=device)
            mask_2[shuffle_idx_2[visible_pattern_count:]] = True
        else:
            # For ablation study, if random masking is specified, then the masked patches will be randomly selected
            shuffle_idx = torch.randperm(total_pattern_count)
            unshuffle_idx = torch.argsort(shuffle_idx)
            mask = torch.zeros(total_pattern_count, dtype=torch.bool, device=device)
            mask[shuffle_idx[visible_pattern_count:]] = True

        # Get reconstruction target
        src_pattern_feat = self.online_pattern_encoder.encode_node(dataset, src_patterns, self.node_raw_features, src_efeat)
        src_pattern_feat = src_pattern_feat.view(h, num_nodes, num_neighbors, -1).mean(0)
        src_pattern_feat = torch.cat([src_pattern_feat, src_temb_2], dim=-1)
        src_target = self.online_transformer_encode(src_pattern_feat).detach()

        # Advanced augmentation strategy
        # Not implemented in TGPM, but you are welcomed to test data augmentation.
        feat_mode = random.choice(['col', 'row'])
        pattern_mode = random.choice(['mask', 'random'])
        if self.aug_data:
            feat_aug, _ = mask_feature(self.node_raw_features, self.mask_feat_ratio, mode=feat_mode, training=True)
            src_patterns_aug, _ = mask_patterns(src_patterns, self.mask_node_ratio, mode=pattern_mode, training=True)
        else:
            feat_aug = self.node_raw_features
            src_patterns_aug = src_patterns

        # Forward with masked pattern
        src_pattern_feat = self.vq_encoder.encode_node(dataset, src_patterns_aug, feat_aug, src_efeat)
        src_pattern_feat = src_pattern_feat.view(h, num_nodes, num_neighbors, -1).mean(0)
        src_pattern_feat = torch.cat([src_pattern_feat, src_temb_2], dim=-1)
        orig_src_pattern = src_pattern_feat

        pattern_mask_tokens = self.mask_token.repeat(num_nodes, masked_pattern_count, 1)
        src_pattern_feat_vis = orig_src_pattern[:, shuffle_idx[:visible_pattern_count], :]
        src_pattern_feat = torch.cat([src_pattern_feat_vis, pattern_mask_tokens], 1)[:, unshuffle_idx, :]

        src_emb_with_mask = self.transformer_encode(src_pattern_feat)
        src_pattern_emb = self.transformer_decode(src_emb_with_mask)
        src_pattern_emb = self.decoder_lin(src_pattern_emb)

        if not self.random_mask:
            src_pattern_feat_vis_2 = orig_src_pattern[:, shuffle_idx_2[:visible_pattern_count], :]
            src_pattern_feat_2 = torch.cat([src_pattern_feat_vis_2, pattern_mask_tokens], 1)[:, unshuffle_idx_2, :]

            src_emb_with_mask_2 = self.transformer_encode(src_pattern_feat_2)
            src_pattern_emb_2 = self.transformer_decode(src_emb_with_mask_2)
            src_pattern_emb_2 = self.decoder_lin(src_pattern_emb_2)

        if self.lt and self.st and not self.random_mask:
            loss_recon = F.mse_loss(src_pattern_emb[:, mask, :], src_target[:, mask, :])
            loss_recon += F.mse_loss(src_pattern_emb_2[:, mask_2, :], src_target[:, mask_2, :])
        elif self.lt or self.random_mask:
            loss_recon = F.mse_loss(src_pattern_emb[:, mask, :], src_target[:, mask, :])
        elif self.st:
            loss_recon = F.mse_loss(src_pattern_emb_2[:, mask_2, :], src_target[:, mask_2, :])
        else:
            loss_recon = 0.
        
        # Code for NTP task
        src_final_emb = self.transformer_encode(orig_src_pattern)
        
        time_list = time_list.view(-1, 1)
        src_pad_tlist = src_pad_tlist.to(device)
        src_time_ref = time_list - src_pad_tlist[:, 1:]
        
        src_time_pred = self.decoder_next_time(src_final_emb[:, :-1, :])
        src_mask = (src_time_ref != time_list)

        # If all the patterns in this batch don't have meaningful tokens, abandon NTP loss.
        if not torch.any(src_mask):
            loss = loss_recon * self.alpha
        else:
            src_time_target = self.time_encoder(src_time_ref)
            loss_ntp = F.mse_loss(src_time_pred[src_mask], src_time_target[src_mask])
            loss = loss_recon * self.alpha + loss_ntp * self.beta

        return loss