"""Reproduce E4 for the saved k=2 union-kNN graph. Run with Python."""
from pathlib import Path
from fractions import Fraction
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import skew

RESULTS = Path(__file__).resolve().parent.parent / 'results'
OUT = RESULTS / 'E4' / '_degreeDistribution'


def table(name, fields, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = RESULTS / 'graphs/k002.npz'
    weighted = sparse.load_npz(source).tocsr()
    weighted.eliminate_zeros()
    assert np.isfinite(weighted.data).all() and (weighted.data > 0).all()
    assert (weighted != weighted.T).nnz == 0
    assert np.all(weighted.diagonal() == 0)
    adjacency = weighted.copy().astype(np.int64)
    adjacency.data[:] = 1
    # Verify saved graph is precisely the union of each node's first two neighbors.
    with np.load(RESULTS / 'analysis_data.npz', allow_pickle=False) as data:
        neighbors = data['neighbors'][:, :2]
    directed = sparse.csr_matrix((np.ones(neighbors.size, dtype=np.int64),
        (np.repeat(np.arange(adjacency.shape[0]), 2), neighbors.ravel())), shape=adjacency.shape)
    assert (adjacency != directed.maximum(directed.T)).nnz == 0
    count, labels = connected_components(adjacency, directed=False)
    groups = sorted([np.flatnonzero(labels == i) for i in range(count)],
                    key=lambda ids: (-len(ids), int(ids[0])))
    with np.load(RESULTS / 'components/k002.npz', allow_pickle=False) as data:
        saved = data['component_ids']
        assert len(np.unique(saved)) == count
        assert all(len(np.unique(saved[ids])) == 1 for ids in groups)

    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 140})
    summaries, node_rows = [], []
    for rank, ids in enumerate(groups, 1):
        a = adjacency[ids][:, ids].tocsr()
        degree = np.diff(a.indptr)
        # (A^2 .* A) row sum counts twice the triangles incident on a vertex.
        twice_triangles = np.asarray((a @ a).multiply(a).sum(axis=1)).ravel()
        triangles = twice_triangles // 2
        denominator = degree * (degree - 1)
        clustering = np.divide(twice_triangles, denominator,
                               out=np.zeros(len(ids)), where=denominator > 0)
        graph = nx.from_scipy_sparse_array(a)
        reference = np.array(list(nx.clustering(graph, weight=None).values()))
        assert np.allclose(clustering, reference, rtol=0, atol=1e-14)
        assert degree.sum() == a.nnz and np.all(degree >= 2)
        assert np.all((clustering >= 0) & (clustering <= 1))
        assert twice_triangles.sum() % 6 == 0
        support = np.arange(degree.min(), degree.max() + 1)
        degree_counts = np.array([(degree == d).sum() for d in support])
        fractions = [Fraction(int(t), int(d)) if d else Fraction(0)
                     for t, d in zip(twice_triangles, denominator)]
        exact = sorted(set(fractions))
        c_counts = np.array([fractions.count(value) for value in exact])
        prefix = f'component_{rank:02d}'
        table(f'{prefix}_degree_pmf.csv', ['degree', 'count', 'probability'],
              zip(support, degree_counts, degree_counts / len(ids)))
        table(f'{prefix}_clustering_pmf.csv', ['exact_fraction', 'clustering_coefficient', 'count', 'probability'],
              [(str(v), float(v), int(n), n / len(ids)) for v, n in zip(exact, c_counts)])
        node_rows.extend((rank, int(i), int(d), int(t), str(f), float(c))
                         for i, d, t, f, c in zip(ids, degree, triangles, fractions, clustering))
        summary = dict(component_rank=rank, nodes=len(ids), edges=a.nnz // 2,
            node_fraction=len(ids) / adjacency.shape[0],
            degree_min=int(degree.min()), degree_max=int(degree.max()),
            degree_mean=float(degree.mean()), degree_median=float(np.median(degree)),
            degree_mode=int(support[np.argmax(degree_counts)]),
            degree_std=float(degree.std()),
            degree_skewness=float(skew(degree, bias=False)) if np.ptp(degree) else None,
            degree_2_fraction=float(np.mean(degree == 2)),
            degree_at_most_4_fraction=float(np.mean(degree <= 4)),
            clustering_mean=float(clustering.mean()), clustering_median=float(np.median(clustering)),
            clustering_zero_count=int(np.sum(clustering == 0)),
            clustering_one_count=int(np.sum(clustering == 1)),
            clustering_zero_fraction=float(np.mean(clustering == 0)),
            clustering_one_fraction=float(np.mean(clustering == 1)),
            clustering_distinct_values=len(exact),
            triangles=int(twice_triangles.sum() // 6),
            transitivity=float(twice_triangles.sum() / denominator.sum()))
        summaries.append(summary)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
        axes[0].bar(support, degree_counts / len(ids), color='#356ca5', width=0.8)
        axes[0].set(xlabel='Degree d', ylabel='Fraction of vertices', title='Degree distribution (exact PMF)',
                    xlim=(support.min() - 0.8, support.max() + 0.8))
        axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        markerline, stemlines, baseline = axes[1].stem([float(v) for v in exact], c_counts / len(ids), basefmt=' ')
        plt.setp(markerline, color='#bd6533', markersize=5)
        plt.setp(stemlines, color='#bd6533', linewidth=1.5)
        axes[1].set(xlabel='Local clustering coefficient C', ylabel='Fraction of vertices',
                    title='Local clustering distribution (exact PMF)', xlim=(-0.04, 1.04))
        for ax in axes:
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(axis='y', alpha=0.2)
            ax.set_axisbelow(True)
            ax.set_ylim(bottom=0)
        fig.suptitle(f'k = 2 union-kNN | Component {rank}: {len(ids):,} vertices, {a.nnz // 2:,} edges', fontsize=14)
        fig.savefig(OUT / f'{prefix}_distributions.png', dpi=220)
        fig.savefig(OUT / f'{prefix}_distributions.svg')
        plt.close(fig)

    table('node_metrics.csv', ['component_rank', 'global_node_index', 'degree',
          'incident_triangles', 'clustering_exact_fraction', 'clustering_coefficient'], node_rows)
    table('component_summary.csv', list(summaries[0]), [list(s.values()) for s in summaries])
    output = dict(k=2, graph='undirected union-kNN; unweighted topology',
        source='../../graphs/k002.npz', source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        total_nodes=adjacency.shape[0], total_edges=adjacency.nnz // 2,
        component_count=count, components=summaries,
        validation=['Symmetric, positive, finite weights; no self-loops',
                    'Topology equals symmetrized first two stored nearest neighbors',
                    'Recomputed components agree with saved component partition',
                    'All local coefficients agree with independent NetworkX calculation',
                    'Handshake identity, coefficient bounds and triangle counts checked'])
    (OUT / 'summary.json').write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    s = summaries[0]
    report = f'''# E4：组合连通性分析（k=2）

## 图和分析范围

读取项目已有的 `results/graphs/k002.npz`，不重新生成特征或近邻。该图为无向、无自环的**并集对称化 kNN 图**：只要任一端把另一端列为最近的两个邻居，就保留这条边。因此 k=2 不代表每个节点的最终度都是 2。

全图包含 {adjacency.shape[0]} 个节点、{adjacency.nnz // 2} 条无向边，两个连通分量大小为 1997 和 3。1997 节点分量占全图 99.85%，是本题分析的大连通分量；3 节点分量作为补充单独给出，不与大分量混合统计。分量编号按节点数降序排列，从 1 开始。

## 定义

本题采用组合图指标，将原始高斯相似度边权转为是否连接的二值邻接矩阵 B。

- 度：`d(v) = sum_u B[v,u]`，即邻居数量，而非边权之和。
- 局部聚类系数：`C(v) = 2 T(v) / (d(v)(d(v)-1))`，其中 T(v) 是包含 v 的三角形数；度小于 2 时定义为 0（本图无此情形）。
- 图中纵轴为节点比例，即经验概率质量函数 PMF。聚类系数按精确有理数归并，因此没有直方图分箱造成的形状偏差。

## 大连通分量的结果

![大连通分量的两种分布](component_01_distributions.png)

| 指标 | 数值 |
|---|---:|
| 节点数 | {s['nodes']} |
| 无向边数 | {s['edges']} |
| 度的最小值 / 最大值 | {s['degree_min']} / {s['degree_max']} |
| 平均度 | {s['degree_mean']:.6f} |
| 度的中位数 / 众数 | {s['degree_median']:g} / {s['degree_mode']} |
| 度的偏度（校正样本偏度） | {s['degree_skewness']:.6f} |
| 度为 2 的节点比例 | {s['degree_2_fraction']:.2%} |
| 度不超过 4 的节点比例 | {s['degree_at_most_4_fraction']:.2%} |
| 平均局部聚类系数 | {s['clustering_mean']:.6f} |
| 局部聚类系数中位数 | {s['clustering_median']:.6f} |
| C=0 的节点数 / 比例 | {s['clustering_zero_count']} / {s['clustering_zero_fraction']:.2%} |
| C=1 的节点数 / 比例 | {s['clustering_one_count']} / {s['clustering_one_fraction']:.2%} |
| 三角形总数 | {s['triangles']} |
| 全局传递性（供参考） | {s['transitivity']:.6f} |

**度分布类型：**离散、单峰且右偏的经验分布。峰值位于 d={s['degree_mode']}，多数节点只有少量邻居，较高的度对应越来越少的节点，观测范围为 {s['degree_min']}–{s['degree_max']}。每个节点主动选择两个邻居，其他节点的选择又可能增加其度，这解释了最小度为 2 及向右延伸的形状。这里采用形状描述；仅凭有限范围的右尾不能认定其服从幂律、指数或泊松分布，也不能据此称为无标度网络。

**局部聚类系数分布类型：**在 [0,1] 上的离散分布，具有显著的零值集中和多个有理数尖峰。C=0 的质量为 {s['clustering_zero_fraction']:.2%}，说明这些节点的邻居之间没有连接；C=1 的质量为 {s['clustering_one_fraction']:.2%}，表示这些节点的邻居形成完全子图。小度节点可取的系数值很少，例如度为 2 时只有 0 和 1，因此尖峰是稀疏图的自然结果。它不宜被描述为连续正态分布，也不在此强行拟合某种参数分布。平均局部聚类系数是逐节点平均，全局传递性则按可形成三角形的邻居对加权，两者并不相同。

## 小分量（补充）

![三节点分量的两种分布](component_02_distributions.png)

剩余 3 个节点形成完整三角形 K3，有 3 条边。所有节点度为 2，所有局部聚类系数为 1；对应分布分别是集中在 d=2 和 C=1 的退化分布（单点分布）。

## 可直接用于作业的英文结论

For the undirected union-symmetrized k=2 nearest-neighbor graph, the connected components contain 1,997 and 3 vertices. We analyze the 1,997-vertex component as the large component using unweighted combinatorial degree and local clustering coefficients. Its degree distribution is discrete, unimodal and right-skewed, with mode {s['degree_mode']}, mean {s['degree_mean']:.4f}, and observed range {s['degree_min']}–{s['degree_max']}. Most vertices have low degree; {s['degree_at_most_4_fraction']:.2%} have degree at most 4. The local clustering distribution is discrete on [0,1], with pronounced mass at zero ({s['clustering_zero_fraction']:.2%}) and several rational-valued spikes; its mean is {s['clustering_mean']:.4f}. These spikes reflect the small vertex degrees and the limited possible numbers of edges among neighbors. These are empirical shape descriptions, not claims of a fitted parametric law. The remaining component is K3, whose degree and clustering distributions are point masses at 2 and 1, respectively.

## 文件与复现

- `component_01_distributions.png/.svg`：大分量的度分布及局部聚类系数分布，PNG 用于查看，SVG 用于无损缩放。
- `component_02_distributions.png/.svg`：小分量的补充图。
- `component_XX_degree_pmf.csv`、`component_XX_clustering_pmf.csv`：分布的精确计数与概率。
- `node_metrics.csv`：每个节点的度、三角形数、聚类系数；global_node_index 对应原始 `results/nodes.jsonl`。
- `component_summary.csv`、`summary.json`：汇总统计、输入文件 SHA-256 及验证项目。
- 分析脚本位于 [`code/analyze_e4.py`](../../../code/analyze_e4.py)：在 `MusicGraphAnalysis` 根目录运行 `python code/analyze_e4.py`，结果输出到 `results/E4/_degreeDistribution`，需要 numpy、scipy、matplotlib、networkx。

已验证图的对称性、无自环、k=2 近邻并集结构、连通分量划分及握手定理，并用 NetworkX 独立核对所有节点的局部聚类系数。
'''
    (OUT / 'E4_report.md').write_text(report, encoding='utf-8')
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
