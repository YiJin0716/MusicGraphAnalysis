"""Validate the analysis, draw report figures, and compile the LaTeX report."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
from scipy import sparse
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
from pypdf import PdfReader
from build_graph import read_data

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
ASSETS = RESULTS / 'report_assets'
PDF = ROOT / 'MusicGraphAnalysis_Report.pdf'
SOURCE = ROOT / 'code/MusicGraphAnalysis_Report.tex'
COLORS = ['#326eae', '#d97932']


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def save_chart(fig, name):
    fig.savefig(ASSETS / f'latex_{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def collect_data():
    source = read_json(ROOT / 'source_data/SOURCE.json')
    source_file = ROOT / 'source_data/tracks.jsonl.gz'
    assert hashlib.sha256(source_file.read_bytes()).hexdigest() == source['sha256']
    nodes, raw, clipped = read_data(source_file)
    assert np.array_equal(raw, np.load(RESULTS / 'features_raw.npy'))
    assert nodes == [json.loads(s) for s in (RESULTS / 'nodes.jsonl').read_text(encoding='utf-8').splitlines()]
    f = np.load(RESULTS / 'features.npy')
    with np.load(RESULTS / 'preprocessing.npz') as p:
        x = ((raw[:, p['keep_mask']] - p['mean'][p['keep_mask']]) /
             p['std'][p['keep_mask']]) * p['group_weights']
        assert np.allclose(f, (x - p['pca_mean']) @ p['pca_components'].T, atol=1e-12)
        retained_variance = float(p['explained_variance_ratio'].sum())
    distances = np.load(RESULTS / 'distances.npy')
    weights = np.load(RESULTS / 'similarities.npy')
    neighbors = np.load(RESULTS / 'neighbors.npy')
    assert np.allclose(distances, squareform(pdist(f)), atol=1e-12)
    sort_distances = distances.copy()
    np.fill_diagonal(sort_distances, np.inf)
    assert np.array_equal(neighbors, np.argsort(sort_distances, axis=1, kind='stable')[:, :64])
    sigma = float(np.median(distances[np.arange(len(f)), neighbors[:, 9]]))
    expected = np.exp(-distances ** 2 / (2 * sigma ** 2))
    np.fill_diagonal(expected, 0)
    assert np.allclose(weights, expected, atol=1e-15)
    connectivity = []
    for k in range(2, 65):
        a = sparse.load_npz(RESULTS / f'graphs/k{k:03d}.npz').tocsr()
        directed = sparse.csr_matrix((weights[np.arange(len(f))[:, None], neighbors[:, :k]].ravel(),
            (np.repeat(np.arange(len(f)), k), neighbors[:, :k].ravel())), shape=a.shape)
        assert np.allclose((a - directed.maximum(directed.T)).data, 0, atol=1e-15)
        count, labels = connected_components(a, directed=False)
        sizes = sorted(np.bincount(labels).tolist(), reverse=True)
        connectivity.append(dict(k=k, n_nodes=a.shape[0], n_edges=a.nnz // 2,
                                 n_components=int(count), largest_component_size=sizes[0]))
    old_connectivity = read_csv(RESULTS / 'connectivity.csv')
    for actual, saved in zip(connectivity, old_connectivity):
        assert all(actual[key] == int(saved[key]) for key in actual)
    e4 = read_json(RESULTS / 'E4/_degreeDistribution/summary.json')
    e5 = read_json(RESULTS / 'E5/summary.json')
    k2_hash = hashlib.sha256((RESULTS / 'graphs/k002.npz').read_bytes()).hexdigest()
    assert k2_hash == e4['source_sha256'] == e5['graph_sha256']
    with np.load(RESULTS / 'E5/component_01_spectral_data.npz') as p:
        spectral = {key: p[key].copy() for key in p.files}
    a = sparse.load_npz(RESULTS / 'graphs/k002.npz').tocsr()
    ids = spectral['global_node_index']
    a = a[ids][:, ids]
    l = sparse.load_npz(RESULTS / 'E5/component_01_L_sym.npz')
    strength = np.asarray(a.sum(axis=1)).ravel()
    inv = sparse.diags(1 / np.sqrt(strength))
    assert np.allclose((l - (sparse.eye(len(ids)) - inv @ a @ inv)).data, 0, atol=1e-12)
    q, ev = spectral['eigenvectors'], spectral['eigenvalues']
    assert np.max(np.linalg.norm(l @ q - q * ev, axis=0)) < 1e-10
    labels = spectral['domain_id']
    for row in e5['domains']:
        members = np.flatnonzero(labels == row['domain'])
        assert len(members) == row['nodes']
        assert connected_components(a[members][:, members], directed=False, return_labels=False) == 1
    upper = np.triu_indices(len(f), 1)
    facts = dict(n_tracks=len(nodes), n_artists=len({n['artist_id'] for n in nodes}),
        raw_dimensions=raw.shape[1], pca_dimensions=f.shape[1], sigma=sigma,
        retained_variance=retained_variance, distinct_feature_vectors=len(np.unique(raw, axis=0)),
        genre_labeled=sum(bool(n['genre']) for n in nodes),
        mood_labeled=sum(bool(n['mood/theme']) for n in nodes),
        instrument_labeled=sum(bool(n['instrument']) for n in nodes),
        pair_count=len(upper[0]),
        distance_quantiles=np.quantile(distances[upper], [0, .25, .5, .75, 1]).tolist(),
        similarity_quantiles=np.quantile(weights[upper], [0, .25, .5, .75, 1]).tolist())
    assert facts['n_tracks'] == 2000 and facts['n_artists'] == 673
    (ASSETS / 'verified_facts.json').write_text(json.dumps(facts, indent=2) + '\n', encoding='utf-8')
    return dict(facts=facts, e4=e4, e5=e5, spectral=spectral, connectivity=connectivity,
                distances=distances, weights=weights, nodes=nodes, raw=raw)



def prepare_figures(data):
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'],
        'mathtext.fontset': 'cm', 'font.size': 10, 'axes.titlesize': 11,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': False, 'pdf.fonttype': 42})
    rows = data['connectivity']
    fig, ax = plt.subplots(figsize=(6.8, 1.9), layout='constrained')
    ax.plot([r['k'] for r in rows], [r['n_components'] for r in rows],
            'o-', color='black', ms=2.5, lw=.8)
    ax.set(xlabel='Number of nearest neighbors k', ylabel='Components',
           yticks=[1, 2], xlim=(1, 65), ylim=(.8, 2.2))
    ax.annotate('1,997 + 3 vertices', (2, 2), (10, 1.9),
                arrowprops={'arrowstyle': '-', 'color': 'gray'})
    save_chart(fig, 'connectivity')

    degree = read_csv(RESULTS / 'E4/_degreeDistribution/component_01_degree_pmf.csv')
    cluster = read_csv(RESULTS / 'E4/_degreeDistribution/component_01_clustering_pmf.csv')
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3), layout='constrained')
    axes[0].bar([int(r['degree']) for r in degree], [float(r['probability']) for r in degree], color='.35')
    axes[0].set(xlabel='Degree', ylabel='Fraction of vertices', title='Degree distribution')
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    marker, stems, base = axes[1].stem([float(r['clustering_coefficient']) for r in cluster],
                                     [float(r['probability']) for r in cluster], basefmt=' ')
    plt.setp(marker, color='black', markersize=3)
    plt.setp(stems, color='black', linewidth=1)
    axes[1].set(xlabel='Local clustering coefficient', ylabel='Fraction of vertices',
                title='Local clustering distribution', xlim=(-.03, 1.03))
    for ax in axes:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylim(bottom=0)
    save_chart(fig, 'distributions')

    curve = read_csv(RESULTS / 'E5/fiedler_vs_k.csv')
    fig, ax = plt.subplots(figsize=(6.6, 2.2), layout='constrained')
    ax.plot([int(r['k']) for r in curve[1:]], [float(r['lambda2']) for r in curve[1:]],
            'o-', color='black', ms=2.5, lw=.8)
    ax.plot([2], [0], marker='x', color='black', ms=5)
    ax.axvline(3, color='.5', ls='--', lw=.7)
    ax.set(xlabel='Number of nearest neighbors k', ylabel=r'$\lambda_2(k)$')
    save_chart(fig, 'fiedler')

    values = data['spectral']['eigenvalues']
    gaps = np.minimum(np.diff(values)[:-1], np.diff(values)[1:])
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.35), layout='constrained')
    axes[0].plot(np.arange(1, 33), values[:32], 'o-', color='black', ms=3, lw=.8)
    axes[0].plot([2], [values[1]], marker='o', markerfacecolor='white', markeredgecolor='black', ms=7)
    axes[0].set(xlabel='Eigenvalue index j', ylabel=r'$\lambda_j$', title='32 smallest eigenvalues')
    axes[1].bar(np.arange(2, 33), gaps, color='.45')
    axes[1].bar([2], [gaps[0]], facecolor='white', edgecolor='black')
    axes[1].set(xlabel='Candidate index j', ylabel='Minimum adjacent gap', title='Two-sided isolation')
    save_chart(fig, 'spectrum')

    coords = data['spectral']['embedding']
    domains = data['spectral']['domain_id']
    fig = plt.figure(figsize=(6.8, 3.0))
    fig.subplots_adjust(left=.02, right=.96, bottom=.14, top=.98, wspace=.10)
    for panel, azimuth in enumerate([-60, 35], 1):
        ax = fig.add_subplot(1, 2, panel, projection='3d')
        for domain in [1, 2]:
            mask = domains == domain
            ax.scatter(*coords[mask].T, s=3, alpha=.65, depthshade=False,
                       color=COLORS[domain - 1], label=f'Domain {domain}')
        ax.set(xlabel=r'$q_2$', ylabel=r'$q_3$', zlabel=r'$q_4$')
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_major_locator(MaxNLocator(3))
            axis.set_tick_params(labelsize=8, pad=0)
        ax.view_init(elev=23, azim=azimuth)
        ax.set_box_aspect((1, 1, 1))
        if panel == 1:
            ax.legend(loc='upper left', fontsize=8, frameon=False, markerscale=2)
    fig.savefig(ASSETS / 'latex_embedding.pdf')
    plt.close(fig)

    rows = read_csv(RESULTS / 'E5/domain_genre_summary.csv')
    domain1 = {r['genre']: r for r in rows if r['domain'] == '1'}
    domain2 = {r['genre']: r for r in rows if r['domain'] == '2'}
    tags = sorted(domain1, key=lambda tag: -float(domain1[tag]['whole_component_fraction']))[:12]
    fig, ax = plt.subplots(figsize=(6.8, 2.0), layout='constrained')
    for i, group in enumerate([domain1, domain2]):
        ax.bar(np.arange(len(tags)) + (i - .5) * .38,
               [float(group[tag]['fraction']) for tag in tags], width=.38,
               color=COLORS[i], label=f'Domain {i+1}')
    ax.set_xticks(np.arange(len(tags)), tags, rotation=35, ha='right')
    ax.set_ylabel('Fraction carrying the tag')
    ax.legend(frameon=False, fontsize=9)
    save_chart(fig, 'genres')


def compile_report():
    engine = shutil.which('pdflatex')
    if not engine:
        raise RuntimeError('pdflatex is required (MiKTeX or TeX Live).')
    build = ROOT / 'tmp/pdfs/latex_build'
    build.mkdir(parents=True, exist_ok=True)
    command = [engine, '-interaction=nonstopmode', '-halt-on-error', '-file-line-error',
               f'-output-directory={build}', str(SOURCE)]
    for _ in range(2):
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, errors='replace')
        if result.returncode:
            raise RuntimeError(result.stdout[-8000:] + result.stderr[-1000:])
    log = (build / 'MusicGraphAnalysis_Report.log').read_text(encoding='utf-8', errors='replace')
    if 'Overfull' in log:
        print('LaTeX layout warnings:', '\n'.join(line for line in log.splitlines() if 'Overfull' in line))
    shutil.copy2(build / 'MusicGraphAnalysis_Report.pdf', PDF)
    reader = PdfReader(PDF)
    text = '\n'.join(page.extract_text() for page in reader.pages)
    for section in ['E1.', 'E2.', 'E3.', 'E4.', 'E5.', 'E7.']:
        assert section in text
    assert 'E6.' not in text
    validation = dict(pages=len(reader.pages), language='English', typesetting='LaTeX article / pdflatex',
        included_questions=['E1', 'E2', 'E3', 'E4', 'E5', 'E7'], figure_count=6,
        pdf_sha256=hashlib.sha256(PDF.read_bytes()).hexdigest(), visual_review='pending')
    (ASSETS / 'report_validation.json').write_text(json.dumps(validation, indent=2) + '\n', encoding='utf-8')
    print(f'Compiled {len(reader.pages)} pages: {PDF}')


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    data = collect_data()
    prepare_figures(data)
    compile_report()


if __name__ == '__main__':
    main()
