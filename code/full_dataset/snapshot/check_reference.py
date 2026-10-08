"""Check blocked methods against the validated 2000-track reference and tie cases."""
import json
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

import reference_build_graph as ref
from run_experiments import ROOT, create_distances, create_similarities, exact_neighbors, write_json


def main():
    nodes, raw, clipped = ref.read_data(ROOT / 'code/pilot_tracks.jsonl.gz')
    with threadpool_limits(limits=4):
        features, params = ref.preprocess(raw, 32)
    expected_indices, expected_weights, expected_sigma, fallback = ref.find_neighbors(features, 64)
    with tempfile.TemporaryDirectory(dir=ROOT / 'logs') as directory:
        path = Path(directory)
        create_distances(features, path / 'distances.npy')
        distances = np.load(path / 'distances.npy', mmap_mode='r')
        actual_indices, actual_distances = exact_neighbors(distances)
        sigma = float(np.median(actual_distances[:, 9]))
        weights = np.exp(-.5*(actual_distances/sigma)**2)
        assert np.array_equal(actual_indices, expected_indices), 'Pilot neighbor indices differ'
        assert np.isclose(sigma, expected_sigma, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(weights, expected_weights, atol=1e-12, rtol=1e-12)
        stats = create_similarities(distances, sigma, path / 'similarities.npy')
        similarities = np.load(path / 'similarities.npy', mmap_mode='r')
        expected_similarities = np.exp(-.5*(cdist(features, features)/sigma)**2)
        np.fill_diagonal(expected_similarities, 0)
        np.testing.assert_allclose(similarities, expected_similarities, atol=1e-12, rtol=1e-12)
        assert np.array_equal(distances, distances.T)
        assert np.array_equal(similarities, similarities.T)
        for k in range(2, 65):
            actual = ref.build_graph(actual_indices, weights, k)
            expected = ref.build_graph(expected_indices, expected_weights, k)
            assert np.array_equal(actual.indptr, expected.indptr)
            assert np.array_equal(actual.indices, expected.indices)
            np.testing.assert_allclose(actual.data, expected.data, atol=1e-12, rtol=1e-12)
        del distances, similarities
    # A tied neighborhood crosses the top-k partition boundary; distinct IDs remain.
    tied = np.zeros((80, 3))
    tied[-1] = [1, 0, 0]
    distances = cdist(tied, tied)
    indices, values = exact_neighbors(distances)
    expected = distances.copy()
    np.fill_diagonal(expected, np.inf)
    assert np.array_equal(indices, np.argsort(expected, kind='stable', axis=1)[:, :64])
    write_json(ROOT / 'logs/reference_validation.json', dict(passed=True, pilot_tracks=len(nodes),
               all_63_graphs_match_reference=True, neighbor_indices_exact_match=True,
               matrices_match=True, boundary_ties_and_self_exclusion=True,
               sigma=sigma, weight_max_abs_error=float(np.max(np.abs(weights-expected_weights)))))
    print('REFERENCE VALIDATION PASSED', flush=True)


if __name__ == '__main__':
    main()
