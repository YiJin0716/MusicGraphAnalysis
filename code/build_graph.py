"""Read bundled acoustic statistics, preprocess, and build union-kNN graphs."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

BASE = Path(__file__).resolve().parent
FEATURE_NAMES = ([f'mfcc.mean.{i:02d}' for i in range(13)]
                 + [f'mfcc.std.{i:02d}' for i in range(13)]
                 + [f'hpcp.mean.{i:02d}' for i in range(36)]
                 + [f'hpcp.std.{i:02d}' for i in range(36)]
                 + ['rhythm.bpm', 'rhythm.onset_rate'])


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
                    + '\n', encoding='utf-8')


def extract_features(acoustic):
    """Select raw statistics; convert MFCC covariance/HPCP variance to std."""
    fields = [('lowlevel.mfcc.mean', (13,)), ('lowlevel.mfcc.cov', (13, 13)),
              ('tonal.hpcp.mean', (36,)), ('tonal.hpcp.var', (36,)),
              ('rhythm.bpm', ()), ('rhythm.onset_rate', ())]
    arrays = []
    for name, shape in fields:
        value = acoustic
        for key in name.split('.'):
            value = value[key]
        array = np.asarray(value, dtype=np.float64)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f'Invalid feature: {name}, expected shape {shape}')
        arrays.append(array)
    clipped = 0
    for index, variance in [(1, np.diag(arrays[1]).copy()), (3, arrays[3].copy())]:
        tolerance = 1e-10 * max(1., float(np.max(np.abs(variance))))
        if np.any(variance < -tolerance):
            raise ValueError(f'Negative variance: {fields[index][0]}')
        clipped += int(np.count_nonzero(variance < 0))
        arrays[index] = np.sqrt(np.maximum(variance, 0))
    return np.concatenate([a.reshape(-1) for a in arrays]), clipped


def read_data(path):
    """One gzip JSONL record: {'node': metadata, 'acoustic': selected raw fields}."""
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]
    records.sort(key=lambda row: row['node']['track_id'])
    track_ids = [r['node']['track_id'] for r in records]
    if not records or len(set(track_ids)) != len(track_ids):
        raise ValueError('Data must contain nonempty, unique track IDs')
    nodes, rows, clipped = [], [], 0
    for i, record in enumerate(records):
        node = dict(record['node'], node_index=i)
        try:
            row, count = extract_features(record['acoustic'])
        except (KeyError, ValueError) as error:
            raise ValueError(f"{node['track_id']}: {error}") from error
        nodes.append(node)
        rows.append(row)
        clipped += count
    return nodes, np.asarray(rows), clipped


def preprocess(raw, dimensions=32):
    """z-score -> inverse-sqrt group scaling -> full-SVD PCA, no whitening."""
    mean, std = raw.mean(axis=0), raw.std(axis=0, ddof=0)
    keep = std > 1e-12
    if not keep.any():
        raise ValueError('All features are constant')
    weights = np.zeros(raw.shape[1])
    for start, end in [(0, 26), (26, 98), (98, 100)]:
        count = int(keep[start:end].sum())
        weights[start:end] = 1 / np.sqrt(count) if count else 0
    weighted = ((raw[:, keep] - mean[keep]) / std[keep]) * weights[keep]
    pca = PCA(svd_solver='full', whiten=False, random_state=521).fit(weighted)
    rank = int(np.count_nonzero(pca.singular_values_ > pca.singular_values_[0] * 1e-10))
    if rank == 0:
        raise ValueError('Feature matrix has zero numerical rank')
    d = min(dimensions, rank)
    features = (weighted - pca.mean_) @ pca.components_[:d].T
    params = dict(mean=mean, std=std, keep_mask=keep, group_weights=weights[keep],
                  pca_mean=pca.mean_, pca_components=pca.components_[:d],
                  explained_variance_ratio=pca.explained_variance_ratio_[:d],
                  explained_variance=pca.explained_variance_[:d],
                  singular_values=pca.singular_values_)
    return features, params


def find_neighbors(features, k_max=64):
    """Exact distances, stable ties by node index, and a fixed Gaussian bandwidth."""
    if not 1 <= k_max < len(features) or len(features) <= 10:
        raise ValueError('Require n > 10 and 1 <= k_max < n')
    distances = squareform(pdist(features, metric='euclidean'))
    sorting = distances.copy()
    np.fill_diagonal(sorting, np.inf)
    ordered = np.argsort(sorting, axis=1, kind='stable')[:, :max(k_max, 10)]
    kth = distances[np.arange(len(features)), ordered[:, 9]]
    sigma = float(np.median(kth))
    fallback = None
    if sigma == 0:
        positive = kth[kth > 0]
        fallback = 'median positive 10th-neighbor distance'
        if not positive.size:
            positive = distances[distances > 0]
            fallback = 'median positive pairwise distance'
        if not positive.size:
            raise ValueError('All distances are zero')
        sigma = float(np.median(positive))
    neighbors = ordered[:, :k_max]
    neighbor_distances = distances[np.arange(len(features))[:, None], neighbors]
    weights = np.exp(-0.5 * (neighbor_distances / sigma)**2)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('Nonfinite/underflowed retained edge weights')
    return neighbors, weights, sigma, fallback


def build_graph(neighbors, neighbor_weights, k):
    """Restore a weighted, undirected union-kNN CSR graph for any stored k."""
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= k <= neighbors.shape[1]:
        raise ValueError('k must be an integer within the stored neighbor range')
    n = len(neighbors)
    directed = sparse.csr_matrix((neighbor_weights[:, :k].ravel(),
        (np.repeat(np.arange(n), k), neighbors[:, :k].ravel())), shape=(n, n))
    return directed.maximum(directed.T).tocsr()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=BASE.parent / 'source_data/tracks.jsonl.gz')
    parser.add_argument('--output', type=Path, default=BASE.parent / 'results/rebuilt')
    parser.add_argument('--dimensions', type=int, default=32)
    parser.add_argument('--k-min', type=int, default=2)
    parser.add_argument('--k-max', type=int, default=64)
    parser.add_argument('--save-graphs', action='store_true', help='Also save each graph as a SciPy NPZ')
    args = parser.parse_args()
    if args.dimensions < 1 or not 1 <= args.k_min <= args.k_max:
        parser.error('Require dimensions >= 1 and 1 <= k_min <= k_max')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output folder is not empty; choose a new --output folder')
    nodes, raw, clipped = read_data(args.data)
    print(f'Read {len(nodes)} tracks; extracted {raw.shape[1]} raw features', flush=True)
    features, params = preprocess(raw, args.dimensions)
    neighbors, weights, sigma, fallback = find_neighbors(features, args.k_max)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / 'features_raw.npy', raw)
    np.savez_compressed(args.output / 'preprocessing.npz', **params)
    np.savez_compressed(args.output / 'analysis_data.npz', features=features,
                        neighbors=neighbors, neighbor_weights=weights)
    write_json(args.output / 'feature_names.json', FEATURE_NAMES)
    with (args.output / 'nodes.jsonl').open('w', encoding='utf-8') as f:
        for node in nodes:
            f.write(json.dumps(node, ensure_ascii=False) + '\n')
    rows = []
    if args.save_graphs:
        (args.output / 'graphs').mkdir()
    for k in range(args.k_min, args.k_max + 1):
        graph = build_graph(neighbors, weights, k)
        count, ids = connected_components(graph, directed=False)
        largest = int(np.bincount(ids).max())
        rows.append(dict(k=k, n_nodes=len(nodes), n_edges=graph.nnz//2,
                         n_components=count, largest_component_size=largest,
                         largest_component_fraction=largest/len(nodes)))
        if args.save_graphs:
            sparse.save_npz(args.output / f'graphs/k{k:03d}.npz', graph)
    write_json(args.output / 'connectivity.json', rows)
    summary = dict(n_nodes=len(nodes), artist_count=len({n['artist_id'] for n in nodes}),
        raw_dimensions=raw.shape[1], retained_dimensions=int(params['keep_mask'].sum()),
        requested_pca_dimensions=args.dimensions, pca_dimensions=features.shape[1],
        cumulative_explained_variance=float(params['explained_variance_ratio'].sum()),
        sigma=sigma, bandwidth_neighbor=10, bandwidth_fallback=fallback,
        clipped_variances=clipped, seed=521, k_min=args.k_min, k_max=args.k_max,
        k_c=next((r['k'] for r in rows if r['n_components']==1), None),
        symmetrization='union', weight='Gaussian similarity, not shortest-path cost')
    write_json(args.output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Output: {args.output.resolve()}')


if __name__ == '__main__':
    main()
