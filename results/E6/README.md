# E6: weighted Leiden communities

For a detailed Chinese analysis, see [report.md](report.md).

The main comparison uses the first connected graph, k=3. The denser k=64 graph shows sensitivity to graph density. Leiden maximizes standard weighted modularity at resolution 1 with the original Gaussian edge weights. Five independent seeds (521–525) are recorded in each `restarts.csv`; the run with the highest modularity is used for the representative partition. Genre, mood/theme, and E5 domains are used only after partitioning to describe the result.

| k | Communities | Community sizes | Weighted modularity | Median restart ARI | ARI against E5 domains |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 16 | 91–210 | 0.7159 | 0.493 | 0.087 |
| 64 | 5 | 267–598 | 0.4646 | 0.766 | 0.263 |

`kXXX/nodes.csv` maps every track to its community; `membership.npy` uses the original zero-based node index. `communities.csv`, `community_features.csv`, and the label tables give community summaries. `e5_overlap.csv` compares Leiden communities with E5's two nodal domains. `metrics.json` records parameters, graph hash, quality, and stability. Figures illustrate community size, label profiles, E5 overlap, and communities in the E5 spectral embedding.

From the repository root, after generating the E5 results described in `results/E4_E5_extension/README.md`, run:

```sh
python -m pip install -r results/E6/requirements.txt
python code/analyze_e6.py
```

The script reads existing graph, feature, and node files from `results/` and writes to `results/E6/`. Community IDs have meaning only within one graph. The two modularity values use different edge sets and null models and should not be ranked as if they measured clustering accuracy. The five-run stability is lower at k=3, so individual small communities should be interpreted cautiously.
