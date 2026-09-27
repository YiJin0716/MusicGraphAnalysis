# E4/E5 extension: k=3 and k=64

These results use the repository's saved 32-dimensional PCA features and weighted union-kNN graphs. The k=2 calculation is included as a baseline; its spectral values describe only the 1,997-node largest component. The k=3 and k=64 graphs each contain all 2,000 tracks.

| k | Edges | Mean degree | Mean clustering | Component λ₂ | E5 domain sizes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 3,308 | 3.313 | 0.140 | 0.018170 | 1,034 / 963 |
| 3 | 4,888 | 4.888 | 0.169 | 0.032154 | 1,083 / 917 |
| 64 | 96,867 | 96.867 | 0.349 | 0.137470 | 1,014 / 986 |

Each `kXXX/` folder contains E4 degree and clustering data, E5 eigenvalues and nodal-domain data, numerical summaries, and selected figures. `spectral_data.npz` stores the 33 smallest eigenpairs, domain membership, and global node indices. `comparison.csv` and `partition_comparison.csv` summarize changes across k; `fiedler_vs_k.csv` contains the full-graph curve for k=2..64. The full graph is disconnected at k=2, so its λ₂ is zero, distinct from the largest component's value above.

From the repository root, regenerate the analysis with:

```sh
python code/analyze_e4_e5_extension.py --run-dir results
python code/compare_e4_e5_extension.py --run-dir results
```

The scripts write to this directory and read the existing files in `results/graphs/`, `results/nodes.jsonl`, `results/features_raw.npy`, and `results/feature_names.json`. The figures shown here are a compact selection; regeneration also produces additional diagnostic images. Domain labels and eigenvector signs are arbitrary across graphs. See `partition_comparison.csv` for track-aligned comparison.
