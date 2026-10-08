"""Independently verify all E1-E3 artifacts, matrices, and 63 saved graphs."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist

from download import digest
from run_experiments import ROOT, write_json


def check(condition, message):
    if not condition:
        raise ValueError(message)


def validate(name):
    start_time = time.time()
    out = ROOT / 'experiments' / name
    config = json.loads((out / 'config.json').read_text(encoding='utf-8'))
    check(digest(ROOT / 'code/run_experiments.py') == config['implementation_sha256'], 'Implementation changed')
    check(digest(ROOT / 'code/reference_build_graph.py') == config['reference_code_sha256'], 'Reference changed')
    check(digest(ROOT / 'data/audit.json') == config['input_audit_sha256'], 'Input audit changed')
    e1, e2, e3 = [json.loads((out / f'E{i}.json').read_text(encoding='utf-8')) for i in range(1, 4)]
    nodes = [json.loads(line) for line in (out / 'nodes.jsonl').read_text(encoding='utf-8').splitlines()]
    n = len(nodes)
    check(n == e1['n_nodes'], 'Node count')
    check([node['node_index'] for node in nodes] == list(range(n)), 'Node alignment')
    ids = [node['track_id'] for node in nodes]
    check(ids == sorted(set(ids)), 'Unique sorted IDs')
    source_nodes = [json.loads(line) for line in (ROOT / 'data/nodes.jsonl').read_text(encoding='utf-8').splitlines()]
    indices = [i for i, node in enumerate(source_nodes) if name == 'full' or node['in_moodtheme']]
    check(ids == [source_nodes[i]['track_id'] for i in indices], 'Dataset subset alignment')
    source_raw = np.load(ROOT / 'data/features_raw.npy', mmap_mode='r')
    raw = np.load(out / 'features_raw.npy')
    features = np.load(out / 'features.npy')
    check(np.array_equal(raw, source_raw[indices]), 'Raw features alignment')
    check(raw.shape == (n, 100) and features.shape == (n, 32), 'Feature shapes')
    check(np.isfinite(raw).all() and np.isfinite(features).all(), 'Finite features')
    with np.load(out / 'preprocessing.npz') as params:
        np.testing.assert_allclose(raw.mean(axis=0), params['mean'], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(raw.std(axis=0), params['std'], rtol=1e-12, atol=1e-12)
        keep = params['keep_mask']
        weighted = ((raw[:, keep]-params['mean'][keep])/params['std'][keep])*params['group_weights']
        reconstructed = (weighted-params['pca_mean']) @ params['pca_components'].T
        np.testing.assert_allclose(features, reconstructed, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(params['pca_components'] @ params['pca_components'].T, np.eye(32), rtol=1e-10, atol=1e-10)
        check(np.isclose(params['explained_variance_ratio'].sum(), e1['cumulative_explained_variance']), 'PCA explained variance')
    with np.load(out / 'analysis_data.npz') as bundle:
        np.testing.assert_array_equal(bundle['features'], features)
        neighbors, weights = bundle['neighbors'], bundle['neighbor_weights']
    neighbor_distances = np.load(out / 'neighbor_distances.npy')
    distances = np.load(out / 'distances.npy', mmap_mode='r')
    similarities = np.load(out / 'similarities.npy', mmap_mode='r')
    check(distances.shape == similarities.shape == (n, n), 'Matrix shapes')
    check(distances.dtype == similarities.dtype == np.float64, 'Matrix precision')
    check(np.all(np.diag(distances) == 0) and np.all(np.diag(similarities) == 0), 'Matrix diagonals')
    check(neighbors.shape == weights.shape == neighbor_distances.shape == (n, 64), 'Neighbor shapes')
    check((neighbors >= 0).all() and (neighbors < n).all() and not np.any(neighbors == np.arange(n)[:, None]), 'Neighbor ranges/self')
    # Sequential full-entry checks. Exact equality to the symmetric Euclidean
    # function and its pointwise Gaussian kernel proves symmetry without random
    # full-column disk reads. cdist uses the same coordinate order for either
    # endpoint order: squared differences are identical float64 values.
    block_size = 128
    hist = np.zeros(100, dtype=np.int64)
    zero_pairs, duplicates = 0, 0
    sigma = e2['sigma']
    for start in range(0, n, block_size):
        end = min(n, start + block_size)
        d = np.array(distances[start:end])
        w = np.array(similarities[start:end])
        expected = cdist(features[start:end], features)
        check(np.array_equal(d, expected), 'Every distance must exactly match the symmetric Euclidean function')
        expected_w = np.exp(-.5*(d/sigma)**2)
        expected_w[np.arange(end-start), np.arange(start, end)] = 0
        check(np.array_equal(w, expected_w), 'Every similarity must exactly match the symmetric Gaussian function')
        np.testing.assert_allclose(d[np.arange(end-start)[:, None], neighbors[start:end]], neighbor_distances[start:end], rtol=1e-12, atol=1e-12)
        sorting = d.copy()
        sorting[np.arange(end-start), np.arange(start, end)] = np.inf
        selected = np.argpartition(sorting, 63, axis=1)[:, :64]
        for local, candidate in enumerate(selected):
            threshold = sorting[local, candidate].max()
            cand = np.flatnonzero(sorting[local] <= threshold)
            ordered = cand[np.lexsort((cand, sorting[local, cand]))][:64]
            check(np.array_equal(ordered, neighbors[start+local]), 'Exact nearest-neighbor order')
            upper = w[local, start+local+1:]
            hist += np.histogram(upper, bins=100, range=(0, 1))[0]
            zero_pairs += int(np.count_nonzero(upper == 0))
            duplicates += int(np.count_nonzero(d[local, start+local+1:] == 0))
        if start // block_size % 40 == 0:
            print(f'Validate {name} matrices + neighbors: {end}/{n}', flush=True)
        # Release all old file mappings, keeping resident mapped pages bounded.
        distances._mmap.close()
        similarities._mmap.close()
        distances = np.load(out / 'distances.npy', mmap_mode='r')
        similarities = np.load(out / 'similarities.npy', mmap_mode='r')
    check(np.array_equal(hist, e2['similarity_histogram']['counts']), 'Similarity histogram')
    check(zero_pairs == e2['zero_similarity_pairs'] and duplicates == e2['zero_distance_pairs'], 'Zero pair counts')
    check(hist.sum() == n*(n-1)//2 == e2['unordered_pairs'], 'Pair coverage')
    check(np.isclose(np.median(neighbor_distances[:, 9]), sigma, rtol=1e-12) or e2['bandwidth_fallback'] is not None, 'Bandwidth')
    check(np.isfinite(weights).all() and (weights > 0).all(), 'Retained weights positive')
    np.testing.assert_allclose(weights, np.exp(-.5*(neighbor_distances/sigma)**2), rtol=1e-12, atol=1e-15)
    distances._mmap.close()
    similarities._mmap.close()
    del distances, similarities
    rows = json.loads((out / 'connectivity.json').read_text(encoding='utf-8'))
    check([row['k'] for row in rows] == list(range(2, 65)), 'k coverage')
    previous = None
    for row in rows:
        k = row['k']
        graph = sparse.load_npz(out / f'graphs/k{k:03d}.npz').tocsr()
        check(graph.shape == (n, n) and graph.nnz == row['n_edges']*2, 'Graph shape/edges')
        check(not np.any(graph.diagonal()) and (graph-graph.T).nnz == 0, 'Graph symmetry/diagonal')
        check(np.isfinite(graph.data).all() and (graph.data > 0).all(), 'Graph weights')
        directed = sparse.coo_matrix((weights[:, :k].ravel(),
                                      (np.repeat(np.arange(n), k), neighbors[:, :k].ravel())), shape=(n, n)).tocsr()
        expected = directed.maximum(directed.T)
        check((graph-expected).nnz == 0, 'Union-kNN graph weights/topology')
        if previous is not None:
            check((previous - previous.multiply(graph.sign())).nnz == 0, 'Nested edge sets')
        previous = graph
        count, labels = connected_components(graph, directed=False)
        with np.load(out / f'components/k{k:03d}.npz') as comp:
            saved_labels, sizes = comp['labels'], comp['sizes']
        check(count == row['n_components'] == len(sizes), 'Component count')
        check(np.array_equal(np.bincount(saved_labels), sizes), 'Component label sizes')
        check(sizes.tolist() == row['component_sizes'] and sizes[0] == row['largest_component_size'], 'Component statistics')
        mapping = np.full(count, -1, dtype=np.int32)
        for original in range(count):
            mapped = np.unique(saved_labels[labels == original])
            check(len(mapped) == 1, 'Component partition mismatch')
            mapping[original] = mapped[0]
        check(len(set(mapping)) == count, 'Merged component labels')
        check(np.isclose(sizes[0]/n, row['largest_component_fraction']), 'Largest fraction')
        print(f'Validate {name} graph k={k}: PASS', flush=True)
    check(e3['k_c'] == next((row['k'] for row in rows if row['n_components']==1), None), 'Connectivity threshold')
    paths = sorted(p for p in out.rglob('*') if p.is_file() and p.name not in ['validation.json', 'artifact_manifest.json', 'COMPLETE.json'])
    artifacts = []
    for index, path in enumerate(paths):
        artifacts.append({'file': str(path.relative_to(out)), 'bytes': path.stat().st_size, 'sha256': digest(path)})
        if index % 30 == 0:
            print(f'Hash {name} artifacts: {index+1}/{len(paths)}', flush=True)
    write_json(out / 'artifact_manifest.json', artifacts)
    validation = dict(passed=True, full_matrix_entries_checked=n*n,
                      matrix_values_match_symmetric_functions_exactly=True,
                      matrix_symmetry_method='Every entry equals the deterministic symmetric Euclidean / Gaussian formula exactly',
                      exact_neighbors_checked=n*64, saved_graphs_checked=63,
                      source_and_node_alignment=True, PCA_reconstruction=True,
                      all_component_labels_checked=True, all_artifacts_sha256=True,
                      elapsed_seconds=time.time()-start_time, validator_sha256=digest(Path(__file__)))
    write_json(out / 'validation.json', validation)
    complete = json.loads((out / 'COMPLETE.json').read_text(encoding='utf-8'))
    complete['independently_validated'] = True
    write_json(out / 'COMPLETE.json', complete)
    print(f'{name} INDEPENDENT VALIDATION PASSED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=['moodtheme', 'full', 'both'], default='both')
    args = parser.parse_args()
    for name in (['moodtheme', 'full'] if args.experiment == 'both' else [args.experiment]):
        validate(name)
