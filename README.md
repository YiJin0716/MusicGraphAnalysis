# MusicGraphAnalysis

Connectivity and spectral analysis of a music similarity graph built from 2,000 MTG-Jamendo tracks.

Read the [English PDF report](MusicGraphAnalysis_Report.pdf) for the dataset, graph construction, degree and clustering distributions, Laplacian spectrum, nodal domains, and discussion (E1-E5 and E7).

## Analysis

- Extract 100 acoustic features covering timbre, pitch class, and rhythm.
- Standardize and balance feature groups, then retain 32 principal components (94.42% of weighted variance).
- Build Gaussian similarities and undirected union-kNN graphs for k=2 through 64.
- Analyze the k=2 graph's 1,997-vertex component; the remaining component is a three-vertex triangle.
- Study the connectivity threshold k=3, the Fiedler curve, and two spectral nodal domains of sizes 963 and 1,034.

## Files

| Path | Contents |
| --- | --- |
| `code/` | Python analysis scripts, dependencies, and LaTeX report source |
| `source_data/` | Acoustic statistics, provenance, citation, and upstream license |
| `results/` | Features, similarities, graph matrices, statistics, and figures |
| `MusicGraphAnalysis_Report.pdf` | Complete English report |

The saved data and graph artifacts are included so the analysis can be inspected without downloading audio.

## Run

From the repository root:

```sh
python -m pip install -r code/requirements.txt
python code/load_graph.py --k 2 --largest-component
python code/analyze_e4.py
python code/analyze_e5.py
python code/build_report.py
```

Report compilation requires `pdflatex` from MiKTeX or TeX Live. The report uses the standard LaTeX `article` class. Analysis outputs go to `results/E4/` and `results/E5/`.

To rebuild features and graphs into a separate, empty output directory:

```sh
python code/build_graph.py --output results/rebuilt --save-graphs
```

See [code/README.md](code/README.md) for input formats, preprocessing details, and artifact descriptions.

## Data source

The tracks form a two-stage pilot sample from the MTG-Jamendo mood/theme feature archives, using seed 521. The repository contains acoustic statistics and metadata, not audio. Sampling details and source hashes are recorded in [source_data/SOURCE.json](source_data/SOURCE.json).

Dataset citation: D. Bogdanov, M. Won, P. Tovstogan, A. Porter, and X. Serra, *The MTG-Jamendo Dataset for Automatic Music Tagging*, ICML Music Discovery Workshop, 2019. See [CITATION.bib](source_data/CITATION.bib), [upstream documentation](source_data/UPSTREAM_README.md), and the accompanying [upstream license](source_data/LICENSE).

## E4/E5 at k=3 and k=64; E6 Leiden communities

The additional analyses reuse the saved graphs and feature data in `results/`. See [selected-k E4/E5 results](results/selected_k/README.md) and [E6 results](results/E6/README.md) for methods, tables, per-track outputs, and figures. The k=2 calculation in the selected-k folder is a baseline for comparison.

To reproduce these results from the repository root:

```sh
python -m pip install -r code/requirements.txt
python -m pip install -r results/E6/requirements.txt
python code/analyze_selected_k.py --run-dir results
python code/compare_selected_k.py --run-dir results
python code/analyze_e6.py
```
