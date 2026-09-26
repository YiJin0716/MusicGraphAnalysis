"""E5: weighted normalized Laplacians, spectral embeddings and nodal domains.

Run: python code/analyze_e5.py (from MusicGraphAnalysis, or use an absolute path).
All generated artifacts go to results/E5; existing graph files are read-only.
"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
OUT = RESULTS / 'E5'
SEED = 521
COLORS = ['#326eae', '#d97932', '#388768', '#9a62ae']


def write_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                      allow_nan=False) + '\n', encoding='utf-8')


def write_csv(name, fields, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def savefig(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=210, bbox_inches='tight')
    fig.savefig(OUT / f'{name}.svg', bbox_inches='tight')
    plt.close(fig)


def load(k):
    a = sparse.load_npz(RESULTS / f'graphs/k{k:03d}.npz').tocsr()
    a.eliminate_zeros()
    assert (a != a.T).nnz == 0 and np.all(a.diagonal() == 0)
    assert np.isfinite(a.data).all() and np.all(a.data > 0)
    return a


def solve(a, number):
    l = laplacian(a, normed=True).tocsr()
    if number >= a.shape[0]:
        values, vectors = np.linalg.eigh(l.toarray())
    else:
        values, vectors = eigsh(l, k=number, sigma=-1e-6, which='LM',
            tol=1e-11, v0=np.random.default_rng(SEED).normal(size=a.shape[0]))
        order = np.argsort(values)
        values, vectors = values[order], vectors[:, order]
    # Resolve the arbitrary sign reproducibly: largest absolute entry is positive.
    for column in range(vectors.shape[1]):
        pivot = np.argmax(np.abs(vectors[:, column]))
        if vectors[pivot, column] < 0:
            vectors[:, column] *= -1
    residuals = np.linalg.norm(l @ vectors - vectors * values, axis=0)
    orth_error = np.max(np.abs(vectors.T @ vectors - np.eye(vectors.shape[1])))
    assert residuals.max() < 1e-8 and orth_error < 1e-8
    assert abs(values[0]) < 1e-8 and values[1] > 1e-8
    assert values.min() > -1e-8 and values.max() <= 2 + 1e-8
    strength = np.asarray(a.sum(axis=1)).ravel()
    null = np.sqrt(strength)
    null /= np.linalg.norm(null)
    assert np.linalg.norm(l @ null) < 1e-10
    assert abs(abs(null @ vectors[:, 0]) - 1) < 1e-8
    return l, values, vectors, residuals, float(orth_error)


def domains(a, vector):
    tolerance = 1e-10 * np.max(np.abs(vector))
    sign = np.where(vector > tolerance, 1, np.where(vector < -tolerance, -1, 0))
    labels = np.full(len(vector), -1, dtype=int)
    rows = []
    next_id = 1
    # Strong nodal domains are connected components of each induced sign graph.
    for s in [-1, 1]:
        ids = np.flatnonzero(sign == s)
        count, local = connected_components(a[ids][:, ids], directed=False)
        for c in sorted(range(count), key=lambda c: -np.sum(local == c)):
            members = ids[local == c]
            labels[members] = next_id
            rows.append(dict(domain=next_id, sign=int(s), nodes=len(members)))
            next_id += 1
    assert np.all(labels > 0), 'Numerical zero entries require a separate boundary treatment.'
    # Check robustness of labels to a much looser numerical zero threshold.
    assert np.min(np.abs(vector)) > 1e-7 * np.max(np.abs(vector))
    return labels, rows, float(tolerance)


def embedding_plot(coords, labels, domain_rows, name, colored=True):
    fig = plt.figure(figsize=(12.2, 5.4), layout='constrained')
    for panel, azimuth in enumerate([-60, 35], 1):
        ax = fig.add_subplot(1, 2, panel, projection='3d')
        if colored:
            for row in domain_rows:
                mask = labels == row['domain']
                ax.scatter(*coords[mask].T, s=8, alpha=0.65, depthshade=False,
                    color=COLORS[(row['domain'] - 1) % len(COLORS)],
                    label=f"Domain {row['domain']} ({row['sign']:+d}), n={row['nodes']}")
        else:
            ax.scatter(*coords.T, s=8, alpha=0.55, color=COLORS[0])
        ax.set(xlabel='$q_2$', ylabel='$q_3$', zlabel='$q_4$')
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_major_locator(MaxNLocator(4))
        ax.view_init(elev=23, azim=azimuth)
        ax.set_box_aspect((1, 1, 1))
        if colored and panel == 1:
            ax.legend(loc='upper left', fontsize=8)
    fig.suptitle('k=2 | Weighted normalized-Laplacian embedding'
                 + (' | Nodal domains of $q_2$' if colored else ''), fontsize=13)
    savefig(fig, name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False})
    metadata = [json.loads(line) for line in (RESULTS / 'nodes.jsonl').read_text(encoding='utf-8').splitlines()]
    raw = np.load(RESULTS / 'features_raw.npy', allow_pickle=False)
    names = json.loads((RESULTS / 'feature_names.json').read_text(encoding='utf-8'))
    assert [node['node_index'] for node in metadata] == list(range(len(metadata)))
    a_full = load(2)
    count, component_ids = connected_components(a_full, directed=False)
    groups = sorted([np.flatnonzero(component_ids == i) for i in range(count)], key=lambda ids: -len(ids))
    assert [len(ids) for ids in groups] == [1997, 3]
    ids = groups[0]
    a = a_full[ids][:, ids].tocsr()
    # Compute lambda_33 as well, so both neighbor gaps of lambda_32 are available.
    l, values, vectors, residuals, orth_error = solve(a, 33)
    sparse.save_npz(OUT / 'component_01_L_sym.npz', l)
    gap_left = values[1:32] - values[:31]
    gap_right = values[2:33] - values[1:32]
    isolation = np.minimum(gap_left, gap_right)
    selected = int(np.argmax(isolation)) + 1  # Zero-based vector index.
    assert selected == 1, 'Update embedding axes/report if the input graph changes.'
    selected_gap = float(isolation[selected - 1])
    assert selected_gap > 1000 * residuals.max()
    labels, domain_rows, zero_tolerance = domains(a, vectors[:, selected])
    assert len(domain_rows) == 2
    coords = vectors[:, 1:4]
    invariant_residual = float(np.linalg.norm(l @ coords - coords * values[1:4]))
    np.savez_compressed(OUT / 'component_01_spectral_data.npz',
        global_node_index=ids, eigenvalues=values, eigenvectors=vectors,
        embedding=coords, domain_id=labels, selected_eigenvalue_index=selected + 1,
        eigenpair_residuals=residuals)
    write_csv('component_01_eigenvalues.csv',
        ['index_1_based', 'eigenvalue', 'gap_to_previous', 'gap_to_next', 'residual', 'plotted'],
        [(i + 1, value, value - values[i - 1] if i else '',
          values[i + 1] - value if i + 1 < len(values) else '', residuals[i], i < 32)
         for i, value in enumerate(values)])
    write_csv('eigenvalue_selection.csv',
        ['index_1_based', 'eigenvalue', 'left_gap', 'right_gap', 'min_neighbor_gap', 'relative_min_gap'],
        [(i + 2, values[i + 1], gap_left[i], gap_right[i], isolation[i], isolation[i] / values[i + 1])
         for i in range(31)])
    write_csv('component_01_node_embedding.csv',
        ['global_node_index', 'track_id', 'artist_id', 'genres', 'q2', 'q3', 'q4', 'nodal_domain'],
        [(int(g), metadata[g]['track_id'], metadata[g]['artist_id'], '|'.join(metadata[g]['genre']),
          *coords[i], int(labels[i])) for i, g in enumerate(ids)])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    axes[0].plot(np.arange(1, 33), values[:32], 'o-', ms=4)
    axes[0].scatter([2], [values[1]], color=COLORS[1], s=60, zorder=4, label=r'Selected $\lambda_2$')
    axes[0].set(xlabel='Eigenvalue index j (1-based)', ylabel=r'$\lambda_j$', title='32 smallest eigenvalues')
    axes[0].legend()
    axes[1].bar(np.arange(2, 33), isolation, color=COLORS[0])
    axes[1].bar([2], [selected_gap], color=COLORS[1])
    axes[1].set(xlabel='Candidate index j', ylabel='min(left gap, right gap)', title='Two-sided eigenvalue isolation')
    for ax in axes:
        ax.grid(axis='y', alpha=0.2)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    savefig(fig, 'component_01_spectrum')
    embedding_plot(coords, labels, domain_rows, 'component_01_embedding', colored=False)
    embedding_plot(coords, labels, domain_rows, 'component_01_nodal_domains')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout='constrained')
    for ax, (x, y) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
        for domain in domain_rows:
            mask = labels == domain['domain']
            ax.scatter(coords[mask, x], coords[mask, y], s=7, alpha=0.55,
                       color=COLORS[domain['domain'] - 1], label=f"Domain {domain['domain']}")
        ax.set(xlabel=f'$q_{x+2}$', ylabel=f'$q_{y+2}$')
    axes[0].legend(fontsize=8)
    fig.suptitle('Nodal-domain colors in the three coordinate projections')
    savefig(fig, 'component_01_embedding_projections')

    # The small component is supplementary: only three eigenvalues exist.
    small_ids = groups[1]
    small = a_full[small_ids][:, small_ids].tocsr()
    small_l, small_values, small_vectors, small_residuals, _ = solve(small, 3)
    sparse.save_npz(OUT / 'component_02_L_sym.npz', small_l)
    np.savez_compressed(OUT / 'component_02_spectral_data.npz', global_node_index=small_ids,
                        eigenvalues=small_values, eigenvectors=small_vectors)
    write_csv('component_02_eigenvalues.csv', ['index_1_based', 'eigenvalue', 'residual'],
              [(i + 1, v, small_residuals[i]) for i, v in enumerate(small_values)])

    # Full-graph connectivity and Fiedler curve for nested union-kNN graphs.
    with np.load(RESULTS / 'analysis_data.npz', allow_pickle=False) as data:
        near, weights = data['neighbors'][:, 0], data['neighbor_weights'][:, 0]
    directed = sparse.csr_matrix((weights, (np.arange(len(near)), near)), shape=a_full.shape)
    k1 = directed.maximum(directed.T)
    k1_components = int(connected_components(k1, directed=False, return_labels=False))
    curve = []
    for k in range(2, 65):
        graph = load(k)
        n_components = int(connected_components(graph, directed=False, return_labels=False))
        if n_components == 1:
            _, ev, _, res, _ = solve(graph, 3)
            fiedler, residual = float(ev[1]), float(res[1])
        else:
            fiedler, residual = 0., None  # lambda_2=0 for disconnected full graph.
        curve.append(dict(k=k, nodes=graph.shape[0], edges=graph.nnz // 2,
            components=n_components, lambda2=fiedler, residual=residual))
        if k % 10 == 0:
            print(f'Fiedler sweep: k={k}/64', flush=True)
    kc = next(row['k'] for row in curve if row['components'] == 1)
    connected = [row for row in curve if row['k'] >= kc]
    decreases = [(left['k'], right['k'], right['lambda2'] - left['lambda2'])
                 for left, right in zip(connected, connected[1:])
                 if right['lambda2'] - left['lambda2'] < -1e-10]
    write_csv('fiedler_vs_k.csv', list(curve[0]), [list(row.values()) for row in curve])
    fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
    ax.plot([r['k'] for r in connected], [r['lambda2'] for r in connected], 'o-', ms=3)
    ax.scatter([2], [0], marker='x', s=55, color=COLORS[1], label='k=2: full graph disconnected')
    ax.axvline(kc, ls='--', color='gray', lw=1, label=f'Connectivity threshold $k_c$={kc}')
    ax.set(xlabel='Number of nearest neighbors k', ylabel=r'Full-graph Fiedler value $\lambda_2(k)$',
           title='Weighted symmetric normalized Laplacian | 2,000 vertices')
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    savefig(fig, 'fiedler_vs_k')

    # Describe the relationship to original data, without using tags to partition.
    x = raw[ids]
    x_mean, x_std = x.mean(axis=0), x.std(axis=0)
    z = np.divide(x - x_mean, x_std, out=np.zeros_like(x), where=x_std > 0)
    between = np.zeros(x.shape[1])
    total = ((x - x_mean) ** 2).sum(axis=0)
    feature_rows, genre_rows = [], []
    tags = sorted({tag for g in ids for tag in metadata[g]['genre']})
    tag_matrix = np.array([[tag in metadata[g]['genre'] for tag in tags] for g in ids])
    global_tag = tag_matrix.mean(axis=0)
    cut_edges = sparse.triu(a, k=1).tocoo()
    crosses = labels[cut_edges.row] != labels[cut_edges.col]
    cut_weight = float(cut_edges.data[crosses].sum())
    strength = np.asarray(a.sum(axis=1)).ravel()
    for row in domain_rows:
        mask = labels == row['domain']
        means = x[mask].mean(axis=0)
        between += mask.sum() * (means - x_mean) ** 2
        row.update(volume=float(strength[mask].sum()),
                   mean_bpm=float(x[mask, names.index('rhythm.bpm')].mean()),
                   median_bpm=float(np.median(x[mask, names.index('rhythm.bpm')])),
                   mean_onset_rate=float(x[mask, names.index('rhythm.onset_rate')].mean()),
                   artist_count=len({metadata[g]['artist_id'] for g in ids[mask]}))
        for j, feature in enumerate(names):
            feature_rows.append([row['domain'], feature, float(means[j]),
                                 float(x[mask, j].std()), float(z[mask, j].mean())])
        for j, tag in enumerate(tags):
            hits = int(tag_matrix[mask, j].sum())
            genre_rows.append([row['domain'], tag, hits, float(hits / mask.sum()),
                               float(global_tag[j]), float(hits / mask.sum() - global_tag[j])])
    eta2 = np.divide(between, total, out=np.zeros_like(total), where=total > 0)
    feature_order = np.argsort(-eta2)
    write_csv('raw_feature_association.csv', ['feature', 'eta_squared'],
              [(names[j], float(eta2[j])) for j in feature_order])
    write_csv('domain_raw_feature_summary.csv',
              ['domain', 'feature', 'raw_mean', 'raw_std', 'standardized_mean_within_large_component'], feature_rows)
    write_csv('domain_genre_summary.csv',
              ['domain', 'genre', 'count', 'fraction', 'whole_component_fraction', 'fraction_difference'], genre_rows)
    write_csv('domain_summary.csv', list(domain_rows[0]), [list(row.values()) for row in domain_rows])
    top = feature_order[:10]
    mean_z = np.array([z[labels == row['domain']].mean(axis=0) for row in domain_rows])
    fig, ax = plt.subplots(figsize=(10, 3.7), layout='constrained')
    limit = float(np.max(np.abs(mean_z[:, top])))
    im = ax.imshow(mean_z[:, top], cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
    ax.set_xticks(np.arange(len(top)), [names[j] for j in top], rotation=35, ha='right')
    ax.set_yticks(np.arange(len(domain_rows)), [f"Domain {r['domain']} (n={r['nodes']})" for r in domain_rows])
    for i in range(len(domain_rows)):
        for j in range(len(top)):
            ax.text(j, i, f'{mean_z[i, top[j]]:.2f}', ha='center', va='center',
                    color='white' if abs(mean_z[i, top[j]]) > 0.65 * limit else 'black')
    ax.set_title('Top 10 original features ranked by between-domain variance fraction')
    fig.colorbar(im, ax=ax, label='Domain mean (z-score)')
    savefig(fig, 'nodal_domains_raw_features')
    top_tags = np.argsort(-global_tag)[:12]
    fig, ax = plt.subplots(figsize=(10, 4.8), layout='constrained')
    for i, row in enumerate(domain_rows):
        mask = labels == row['domain']
        ax.bar(np.arange(len(top_tags)) + (i - 0.5) * 0.37,
               tag_matrix[mask][:, top_tags].mean(axis=0), width=0.37,
               color=COLORS[i], label=f"Domain {row['domain']}")
    ax.set_xticks(np.arange(len(top_tags)), [tags[j] for j in top_tags], rotation=35, ha='right')
    ax.set(ylabel='Fraction of vertices carrying the tag',
           title='12 most common genre tags | Multi-label metadata')
    ax.legend()
    ax.grid(axis='y', alpha=0.2)
    savefig(fig, 'nodal_domains_genres')

    # Rank tag differences only when the tag has >=30 observations.
    prevalence = np.array([tag_matrix[labels == row['domain']].mean(axis=0) for row in domain_rows])
    eligible = np.flatnonzero(tag_matrix.sum(axis=0) >= 30)
    tag_order = eligible[np.argsort(-np.abs(prevalence[0, eligible] - prevalence[1, eligible]))]
    top_tag_rows = [dict(genre=tags[j], domain1_fraction=float(prevalence[0, j]),
                        domain2_fraction=float(prevalence[1, j])) for j in tag_order[:6]]
    summary = dict(k=2, weighting='original Gaussian similarity weights',
        laplacian='I - D^(-1/2) A D^(-1/2); D is weighted degree (strength)',
        graph_sha256=hashlib.sha256((RESULTS / 'graphs/k002.npz').read_bytes()).hexdigest(),
        component_sizes=[len(g) for g in groups], large_component_edges=a.nnz // 2,
        eigenvalues_plotted=32, eigenvalues_computed=33,
        selected_index_1_based=selected + 1, selected_eigenvalue=float(values[selected]),
        selected_left_gap=float(gap_left[selected-1]), selected_right_gap=float(gap_right[selected-1]),
        selected_min_gap=selected_gap, selected_relative_gap=selected_gap / float(values[selected]),
        selection_rule='Maximize minimum of left and right adjacent gaps among lambda_2,...,lambda_32',
        max_eigenpair_residual=float(residuals.max()), orthogonality_max_error=orth_error,
        embedding_invariance_residual=invariant_residual,
        nodal_zero_tolerance=zero_tolerance, min_abs_selected_eigenvector=float(np.abs(vectors[:, selected]).min()),
        domains=domain_rows, crossing_edges=int(crosses.sum()), crossing_weight=cut_weight,
        normalized_cut=float(sum(cut_weight / row['volume'] for row in domain_rows)),
        k1_components=k1_components, connectivity_threshold=kc,
        fiedler_at_k3=connected[0]['lambda2'], fiedler_at_k64=connected[-1]['lambda2'],
        fiedler_decreases=decreases, small_component_eigenvalues=small_values.tolist(),
        top_raw_features=[dict(feature=names[j], eta_squared=float(eta2[j])) for j in feature_order[:10]],
        largest_genre_prevalence_differences=top_tag_rows,
        versions=dict(numpy=np.__version__, scipy=__import__('scipy').__version__, matplotlib=matplotlib.__version__))
    write_json('summary.json', summary)
    write_report(summary, domain_rows, values, top_tag_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def write_report(s, rows, values, tag_rows):
    domain_table = '\n'.join(
        f"| {r['domain']} | {r['sign']:+d} | {r['nodes']} | {r['mean_bpm']:.2f} | {r['mean_onset_rate']:.3f} |"
        for r in rows)
    feature_table = '\n'.join(f"| {f['feature']} | {f['eta_squared']:.4f} |" for f in s['top_raw_features'])
    genre_table = '\n'.join(f"| {r['genre']} | {r['domain1_fraction']:.2%} | {r['domain2_fraction']:.2%} |" for r in tag_rows)
    trend = ('在 k=3 至 64 的全部相邻取值之间均递增' if not s['fiedler_decreases'] else
             f"总体趋势见图，其中有 {len(s['fiedler_decreases'])} 次相邻 k 的下降，详见 summary.json")
    text = f'''# E5：Laplacian spectral analysis

## 1. 图与归一化拉普拉斯矩阵

分析采用 **k=2、32 维 PCA 后的并集对称化 kNN 图**。原图有 2000 个节点，连通分量大小为 1997 和 3；本题的大连通分量为 1997 节点、{s['large_component_edges']} 条无向边的分量。

E5 使用原图的高斯相似度权重 A，而 E4 的组合度和聚类系数只使用连接结构。令 `D_ii = sum_j A_ij`，构造对称归一化拉普拉斯矩阵：

`L_sym = I - D^(-1/2) A D^(-1/2)`。

D 是加权度（strength），不能替换成无权邻居数量。该分量无孤立节点，L_sym 对称半正定，特征值位于 [0,2]。全文按 `lambda_1 <= lambda_2 <= ...` 从 1 开始编号，`lambda_1 = 0`；对应零特征向量正比于 `sqrt(D_ii)`。矩阵保存为 `component_01_L_sym.npz`。

## 2. 连通阈值与 Fiedler 值随 k 的变化

![Fiedler 曲线](fiedler_vs_k.png)

k=1 时有 {s['k1_components']} 个连通分量；k=2 有两个分量；k=3 全图连通。固定近邻顺序下，并集 kNN 边集随 k 嵌套，因此最小连通阈值为 **k_c=3**。对已有的每张 k=3,...,64 图，用相同高斯边权定义计算全体 2000 个节点的 L_sym 和第二小特征值。

- `lambda_2(3) = {s['fiedler_at_k3']:.8f}`。
- `lambda_2(64) = {s['fiedler_at_k64']:.8f}`。
- 曲线{trend}。更大的谱隙表示在归一化谱意义下，图更不容易被低权重连接分开。
- 这是对本数据的数值观察；归一化拉普拉斯的第二特征值不具有随加边必然单调增加的一般保证，因为 D 也同时变化。

**区别：**k=2 的全图不连通，所以全图 `lambda_2=0`；下面的 `{values[1]:.8f}` 是其 1997 节点连通分量内部的 Fiedler 值，两者不可混用。

## 3. 三维不变子空间嵌入

![三维嵌入](component_01_embedding.png)

选择三个最小非零特征值对应的正交单位特征向量 `q_2,q_3,q_4`，将节点 i 映射为 `(q_2(i),q_3(i),q_4(i))`。三个特征值分别为 {values[1]:.8f}、{values[2]:.8f}、{values[3]:.8f}。

令 `Q=[q_2 q_3 q_4]`，则 `L_sym Q = Q diag(lambda_2,lambda_3,lambda_4)`，所以这三个向量张成不变子空间。图中以单位正交特征向量的分量作为节点坐标。两个视角展示同一嵌入；CSV 保存每个节点的坐标和原始 node_index。

## 4. 最小的 32 个特征值及单重特征值选择

![低端谱与两侧谱隙](component_01_spectrum.png)

取 `ell=32`，图中展示 `lambda_1,...,lambda_32`。第 33 个特征值用于计算第 32 个候选的右侧谱隙。

特征值选择规则为：对 j=2,...,32，计算 `g_j=min(lambda_j-lambda_(j-1), lambda_(j+1)-lambda_j)`，选取 g_j 最大者。本图选到 **lambda_2={values[1]:.10f}**：

- 左侧谱隙为 {s['selected_left_gap']:.10f}。
- 右侧谱隙为 {s['selected_right_gap']:.10f}。
- 最小相邻谱隙与自身之比为 {s['selected_relative_gap']:.2%}。
- 最大特征对残差为 {s['max_eigenpair_residual']:.2e}，比上述谱隙小很多，支持该特征值在数值上为单重且与其余谱分离。

这里“分离良好”按候选谱隙及数值误差量化，不是假定存在通用阈值。选择第二特征向量满足题目要求，并有直接的图分割解释。

## 5. 按节点域划分并着色

![节点域三维着色](component_01_nodal_domains.png)

![二维投影辅助查看](component_01_embedding_projections.png)

节点域定义为 `q_2>0` 和 `q_2<0` 各自诱导子图的**连通分量**。两个诱导子图均连通，因此该划分包含两个强节点域。数值零阈值为 {s['nodal_zero_tolerance']:.2e}，最小绝对坐标为 {s['min_abs_selected_eigenvector']:.2e}，没有零值边界节点；提高阈值后的符号也稳定。

| 节点域 | q2 符号 | 节点数 | 平均 BPM | 平均 onset rate |
|---|---:|---:|---:|---:|
{domain_table}

着色图以选中特征向量 q_2 及特征向量 q_3、q_4 为三个坐标轴。特征向量整体乘以 -1 只会交换域的符号，不改变划分。特征向量的符号约定为最大绝对值对应的分量取正值。

两个域之间共有 {s['crossing_edges']} 条无向边，边权和为 {s['crossing_weight']:.6f}；此符号划分的 normalized cut 为 {s['normalized_cut']:.6f}。两个域都是原图的连通子图，但彼此之间仍有跨域边；它们不是原图原有的两个连通分量。

## 6. 节点域与原始歌曲数据的关系

原始数据比较使用 PCA 前的 100 个声学特征及多标签 genre。节点域由声学特征构建的图确定，genre 标签用于描述各域的音乐类别组成。

![原始声学特征](nodal_domains_raw_features.png)

用 `eta^2 = sum_g n_g (mean_g - mean_all)^2 / sum_i (x_i - mean_all)^2` 衡量每个原始特征的总变异中有多少可由两个域的均值差异描述。它在 [0,1] 内，是描述性效应大小。排名前十的特征为：

| 原始特征 | eta² |
|---|---:|
{feature_table}

热图展示大分量内部标准化后的域均值，正负分别表示高于或低于大分量均值。MFCC 特征与音色/谱包络有关，HPCP 特征描述音高类别分布；具体均值与方向见热图和 CSV。域间的 BPM 与 onset rate 均值见上表，可用于描述节奏差异。

域间差异最明显的特征是 **rhythm.onset_rate**，eta²={s['top_raw_features'][0]['eta_squared']:.4f}：域 1 均值为 {rows[0]['mean_onset_rate']:.3f}，域 2 为 {rows[1]['mean_onset_rate']:.3f}，说明域 2 的起音事件更密集。但域 2 的平均 BPM 反而较低（{rows[1]['mean_bpm']:.2f} 对 {rows[0]['mean_bpm']:.2f}）；起音事件密度与节拍速度是不同指标，不能将这一划分简单概括为“快歌/慢歌”。多个 MFCC 特征也有明显域间差异，说明划分同时反映了声学谱特征。

![常见音乐标签](nodal_domains_genres.png)

图中展示大分量最常见的 12 个 genre 标签。为避免将极少见标签的比例波动过度解读，下表只在至少出现 30 次的标签中，选取两域比例差绝对值最大的六项：

| 标签 | 域 1 占比 | 域 2 占比 |
|---|---:|---:|
{genre_table}

具体而言，域 1 中 classical、soundtrack、ambient 和 orchestral 的占比更高；域 2 中 electronic 和 pop 的占比更高。这与声学统计量上的组间差异一起，为谱节点域提供了可解释的数据联系。

每首歌可有多个 genre 标签，因此同一域的标签比例之和不一定为 100%。上述差异说明节点域与原始声学特征及部分音乐标签有关，但两个域不等价于纯粹的音乐流派分类。声学特征本来就用于建图，其关联不是独立验证；标签比较也是同一数据上的描述性观察，不能证明因果或外部泛化效果。

## 7. 三节点小分量（补充）

小分量未列为 large component。仍保存其 3×3 归一化拉普拉斯矩阵及全部特征对，特征值为 {', '.join(f'{v:.8f}' for v in s['small_component_eigenvalues'])}。它的连接结构是 K3，但原始边权不必相等，所以不能直接套用无权 K3 的两个 1.5 特征值。该分量只有两个非零特征向量，无法构造三个非零方向的三维谱嵌入，也没有 32 个特征值；主分析因此针对大分量。

## 文件与复现

代码位于 [`../../code/analyze_e5.py`](../../code/analyze_e5.py)。在 `MusicGraphAnalysis` 根目录运行：

```powershell
python code/analyze_e5.py
```

输出均位于 `results/E5`。PNG 便于查看，SVG 可无损缩放。`component_01_L_sym.npz` 是 SciPy 稀疏矩阵；`component_01_spectral_data.npz` 含节点索引、33 个特征对、三维坐标和节点域。`component_01_node_embedding.csv` 可将节点域追溯到原歌曲。`fiedler_vs_k.csv`、`eigenvalue_selection.csv` 和原始特征/标签统计表保存完整数值。`summary.json` 记录输入图哈希、软件版本和验证误差。

验证包括：矩阵对称及正权、无自环、零特征向量、特征对残差、正交性、谱范围、三维子空间不变性，以及节点域的连通性和数值符号稳定性。
'''
    (OUT / 'E5_report.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
