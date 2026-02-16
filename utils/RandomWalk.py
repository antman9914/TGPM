from numba import njit
from scipy.sparse import csr_matrix
from pathlib import Path
import os.path as osp
import numpy as np
import torch

@njit(nogil=True)
def _neighbors(indptr, indices_or_data, t):
    return indices_or_data[indptr[t] : indptr[t + 1]]

@njit(nogil=True)
def _random_walk(indptr, indices, walk_length, t, tar, bans):

    walk = np.full(walk_length, -1, dtype=indices.dtype)
    walk[0] = t
    raw_neighbors = _neighbors(indptr, indices, t)
    if len(bans) != 0:
        raw_neighbors = raw_neighbors[raw_neighbors != bans[0]]
    if raw_neighbors.shape[0] == 0:
        return walk
    walk[1] = np.random.choice(raw_neighbors)
    bans = [walk[0]]
    for j in range(2, walk_length):
        if indptr[walk[j - 1]] >= indptr[walk[j - 1] + 1]:
            break
        if walk[j-1] == tar:
            walk[j] = walk[j-1]
            continue
        neighbors = _neighbors(indptr, indices, walk[j - 1])
        mask = np.ones_like(neighbors, dtype=np.bool_)
        for blocker in bans:
            mask = (neighbors != blocker) & mask
        neighbors = neighbors[mask]
        bans.append(walk[j-1])
        if neighbors.shape[0] == 0:
            break
        walk[j] = np.random.choice(neighbors)
    return walk


@njit(nogil=True)
def _random_walk_weighted(indptr, indices, data, eids, walk_length, endpoints, target_time, init_eid, causal_path=False):

    start, t = endpoints
    raw_neighbors = _neighbors(indptr, indices, t)
    raw_time = _neighbors(indptr, data, t)
    init_eids = _neighbors(indptr, eids, t)
    sorted_indices = np.argsort(raw_time)
    raw_neighbors, raw_time = raw_neighbors[sorted_indices], raw_time[sorted_indices]
    init_eids = init_eids[sorted_indices]
    walk = np.full(walk_length + 1, 0, dtype=indices.dtype)
    retrieved_eids = np.full(walk_length, 0, dtype=indices.dtype)
    walk[0] = start
    walk[1] = t
    retrieved_eids[0] = init_eid
    ind = np.searchsorted(raw_time, target_time)
    cur_neighbors = raw_neighbors[:ind]
    cur_time = raw_time[:ind]
    cur_eids = init_eids[:ind]
    if cur_neighbors.shape[0] == 0:
        walk[2:] = t
        return walk, retrieved_eids, 1
    target_time = raw_time[ind-1]
    cur_weights = np.cumsum(np.exp(cur_time - target_time))
    cur_weights = cur_weights / cur_weights[-1]
    ind = np.searchsorted(cur_weights, np.random.rand())
    walk[2] = cur_neighbors[ind]
    retrieved_eids[1] = cur_eids[ind]
    # If no causal path introduced, the target time will be fixed. 
    # Otherwise, the target time will be the timestamp of last sampled element.
    if causal_path:
        target_time = cur_time[ind]
    for j in range(3, walk_length + 1):
        if indptr[walk[j - 1]] >= indptr[walk[j - 1] + 1]:
            break
        neighbors = _neighbors(indptr, indices, walk[j - 1])
        times = _neighbors(indptr, data, walk[j - 1])
        cur_eids = _neighbors(indptr, eids, walk[j - 1])
        sorted_indices = np.argsort(times)
        neighbors, times, cur_eids = neighbors[sorted_indices], times[sorted_indices], cur_eids[sorted_indices]
        ind = np.searchsorted(times, target_time)
        neighbors, times, cur_eids = neighbors[:ind], times[:ind], cur_eids[:ind]
        if neighbors.shape[0] == 0:
            walk[j:] = walk[j-1]
            break
        weights = np.cumsum(np.exp(times - target_time))
        weights = weights / weights[-1]
        ind = np.searchsorted(weights, np.random.rand())
        walk[j] = neighbors[ind]
        retrieved_eids[j-1] = cur_eids[ind]
        if causal_path:
            target_time = times[ind]
    return walk, retrieved_eids, 0


class RandomWalkGraph:
    def __init__(self, num_nodes: int, src: np.ndarray, dst: np.ndarray, data: np.ndarray = None):
        # data is the timestamps
        self.num_nodes = num_nodes
        if data is None:
            self.is_weighted = False
            data = np.ones(len(src), dtype=bool)
        else:
            self.is_weighted = True
        
        edges = csr_matrix((data, (src, dst)), shape=(num_nodes, num_nodes))
        self.eids = np.arange(data.shape[0])
        self.indptr = edges.indptr
        self.indices = edges.indices
        self.data = data

    def generate_random_walk(self, walk_length, start, start_time, init_eid, causal_path=False):
        if self.is_weighted:
            walk, eids, empty = _random_walk_weighted(
                self.indptr, self.indices, self.data, self.eids, walk_length, start, start_time, init_eid, causal_path
            )
        else:
            walk = _random_walk(self.indptr, self.indices, walk_length, start)
        return walk, eids, empty


def get_sentences(dataset, num_nodes, src, dst, data, num_walks, walk_length, p, q, causal_path=False):
    pattern_dir = osp.join('patterns', dataset)
    if causal_path:
        pattern_file = osp.join(pattern_dir, f"pt_{num_walks}_{walk_length}_{p}_{q}_causal.pt")
        eid_file = osp.join(pattern_dir, f"eid_{num_walks}_{walk_length}_{p}_{q}_causal.pt")
    else:
        pattern_file = osp.join(pattern_dir, f"pt_{num_walks}_{walk_length}_{p}_{q}.pt")
        eid_file = osp.join(pattern_dir, f"eid_{num_walks}_{walk_length}_{p}_{q}.pt")
    empty_walk = 0
    if osp.exists(pattern_file) and osp.exists(eid_file):
        patterns = torch.load(pattern_file, map_location=torch.device('cpu'))
        all_eids = torch.load(eid_file, map_location=torch.device('cpu'))
    else:
        src_for_rw = np.concatenate([src, dst], axis=-1)
        dst_for_rw = np.concatenate([dst, src], axis=-1)
        data_for_rw = np.concatenate([data, data], axis=-1)
        walk_graph = RandomWalkGraph(num_nodes, src_for_rw, dst_for_rw, data_for_rw)
        patterns = []
        all_eids = []
        tag = num_walks
        while tag > 0:
            tag -= 1
            ps, es = [], []
            cur_eid = 0
            for nid, start, time in zip(dst, src, data):
                walk_sample, eids, empty = walk_graph.generate_random_walk(walk_length, start=(start, nid), start_time=time, init_eid=cur_eid, causal_path=causal_path)
                eids = eids % dst.shape[0]
                eids = eids + 1
                cur_eid += 1
                ps.append(walk_sample)
                es.append(eids)
                empty_walk += empty
            ps, es = np.stack(ps, axis=0), np.stack(es, axis=0)
            patterns.append(ps)
            all_eids.append(es)
        patterns = np.stack(patterns, axis=0)
        all_eids = np.stack(all_eids, axis=0)
        empty = np.zeros((num_walks, 1, walk_length + 1), dtype=patterns.dtype)
        empty_eid = np.zeros((num_walks, 1, walk_length), dtype=all_eids.dtype)
        # Insert a placeholder for empty token and pattern.
        patterns = np.concatenate([empty, patterns], axis=1)
        all_eids = np.concatenate([empty_eid, all_eids], axis=1)
        patterns, all_eids = torch.from_numpy(patterns), torch.from_numpy(all_eids)

        path = Path(pattern_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(patterns, pattern_file)
        torch.save(all_eids, eid_file)
        print("Total number of empty walk: %d" % empty_walk)
    return patterns, all_eids
    