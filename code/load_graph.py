"""Portable access to the experiment. Run this file for a checked loading example."""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import laplacian

BASE = Path(__file__).resolve().parent

def load_graph(k=32, largest_component=False):
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 2 <= k <= 64:
        raise ValueError('k must be an integer from 2 through 64')
    run = BASE.parent / 'results'
    nodes = [json.loads(s) for s in (run / 'nodes.jsonl').read_text(encoding='utf-8').splitlines()]
    if [n['node_index'] for n in nodes] != list(range(len(nodes))):
        raise ValueError('Node order is not aligned')
    A = sparse.load_npz(run / f'graphs/k{k:03d}.npz').tocsr()
    F = np.load(run / 'features.npy', allow_pickle=False, mmap_mode='r')
    with np.load(run / f'components/k{k:03d}.npz', allow_pickle=False) as p:
        ids = p['component_ids'].copy()
    if A.shape != (len(nodes), len(nodes)) or F.shape[0] != len(nodes):
        raise ValueError('Array dimensions do not match nodes')
    global_indices = np.flatnonzero(ids == 0) if largest_component else np.arange(len(nodes))
    A = A[global_indices][:, global_indices].tocsr()
    return dict(run=run, A=A, features=F[global_indices],
                nodes=[nodes[int(i)] for i in global_indices],
                global_indices=global_indices, component_ids=ids[global_indices],
                degree=np.diff(A.indptr), strength=np.asarray(A.sum(axis=1)).ravel(),
                L=laplacian(A, normed=False).tocsr(),
                L_sym=laplacian(A, normed=True).tocsr())

def export_edges(data, output):
    """One record per undirected edge; source/target use GLOBAL node_index."""
    edges = sparse.triu(data['A'], k=1, format='coo')
    R = np.load(data['run'] / 'distances.npy', allow_pickle=False, mmap_mode='r')
    ix = data['global_indices']
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as f:
        for i, j, w in zip(edges.row, edges.col, edges.data):
            source, target = int(ix[i]), int(ix[j])
            f.write(json.dumps(dict(source=source, target=target, weight=float(w),
                                    distance=float(R[source, target])), ensure_ascii=False) + '\n')
    return int(edges.nnz)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--k', type=int, default=32)
    parser.add_argument('--largest-component', action='store_true')
    parser.add_argument('--edges', type=Path, help='Optional JSONL edge output path')
    args = parser.parse_args()
    data = load_graph(args.k, args.largest_component)
    result = dict(k=args.k, nodes=len(data['nodes']), edges=data['A'].nnz // 2,
                  features_shape=list(data['features'].shape),
                  degree_min=int(data['degree'].min()), degree_max=int(data['degree'].max()),
                  laplacian_shape=list(data['L_sym'].shape))
    if args.edges:
        result['exported_edges'] = export_edges(data, args.edges)
    print(json.dumps(result, ensure_ascii=False, indent=2))
