"""Complete the selected-k comparison: full Fiedler sweep and aligned nodal domains."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.sparse.linalg import eigsh
from sklearn.metrics import adjusted_rand_score

def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)

def fiedler_sweep(run, out):
    rows = []
    for k in range(2, 65):
        a = sparse.load_npz(run / "graphs" / f"k{k:03d}.npz").tocsr()
        count = connected_components(a, directed=False, return_labels=False)
        if count == 1:
            l = laplacian(a, normed=True).tocsr()
            vals, vecs = eigsh(l, k=2, sigma=-1e-6, which="LM", tol=1e-9,
                               v0=np.random.default_rng(521).normal(size=a.shape[0]))
            order = np.argsort(vals)
            vals, vecs = vals[order], vecs[:, order]
            residual = float(np.linalg.norm(l @ vecs[:, 1] - vals[1] * vecs[:, 1]))
            assert residual <= 1e-7 and vals[1] > 0
            lam = float(vals[1])
        else:
            lam, residual = 0.0, None
        rows.append((k, int(count), lam, residual))
        if k % 10 == 0:
            print(f"Fiedler sweep k={k}/64", flush=True)
    write_csv(out / "fiedler_vs_k.csv",
              ["k", "components", "lambda2_full_graph", "residual"], rows)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    connected = [r for r in rows if r[1] == 1]
    ax.plot([r[0] for r in connected], [r[2] for r in connected],
            "o-", ms=2.5, color="#2869aa")
    ax.scatter([2], [0], color="#d87532", marker="x", label="k=2: disconnected")
    for k in (3, 64):
        ax.scatter([k], [rows[k - 2][2]], s=55, color="#d87532")
        ax.annotate(f"k={k}", (k, rows[k - 2][2]), xytext=(5, 6),
                    textcoords="offset points")
    ax.set(xlabel="k", ylabel="Full-graph Fiedler value",
           title="Weighted normalized Laplacian | n=2000 | PCA=32")
    ax.legend()
    fig.savefig(out / "fiedler_vs_k.png", dpi=190, bbox_inches="tight")
    plt.close(fig)
    return rows

def compare_partitions(out):
    partitions = {}
    for k in (2, 3, 64):
        folder = out / f"k{k:03d}"
        with np.load(folder / "spectral_data.npz", allow_pickle=False) as data:
            ids, domains = data["global_node_index"], data["domain_id"]
        info = json.loads((folder / "domain_artist_summary.json").read_text(encoding="utf-8"))
        low_domain = min(info, key=lambda row: row["mean_onset_rate"])["domain_id"]
        partitions[k] = {int(g): int(d == low_domain) for g, d in zip(ids, domains) if d > 0}
    rows = []
    for ka, kb in ((2, 3), (2, 64), (3, 64)):
        common = sorted(partitions[ka].keys() & partitions[kb].keys())
        a = np.array([partitions[ka][i] for i in common])
        b = np.array([partitions[kb][i] for i in common])
        rows.append((ka, kb, len(common), float(np.mean(a == b)),
                     float(adjusted_rand_score(a, b)),
                     int(np.sum((a == 1) & (b == 1))),
                     int(np.sum((a == 1) & (b == 0))),
                     int(np.sum((a == 0) & (b == 1))),
                     int(np.sum((a == 0) & (b == 0)))))
    write_csv(out / "partition_comparison.csv",
              ["k_a", "k_b", "common_nodes", "low_high_agreement", "adjusted_rand_index",
               "both_low", "a_low_b_high", "a_high_b_low", "both_high"], rows)
    return rows

def report(out, summaries, comparisons):
    report = [
        "# k=2、3、64 的 E4/E5 比较", "",
        "使用同一批 2,000 首歌曲、同一 32 维 PCA 表示及固定 Gaussian 带宽；仅改变 union-kNN 的 k。",
        "k=2 分析 1,997 节点最大分量；k=3 和 64 分析各自 2,000 节点的连通全图。",
        "因此 k=2 的分量谱与 k>=3 的全图谱基于不同节点集合，不将其连成同一 Fiedler 曲线。",
        "", "## E4：连接结构", "",
        "| k | 节点 | 边 | 平均度 | 最大度 | 平均局部聚类系数 | C=0 比例 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        k, e = s["k"], s["e4"]
        report.append(f"| {k} | {e['nodes']} | {e['edges']} | {e['degree_mean']:.3f} | {e['degree_max']} | {e['clustering_mean']:.3f} | {e['clustering_zero_fraction']:.2%} |")
    report += ["", "k 增加时，这三张图的边数和平均度明显增加；局部聚类系数平均值也增大。",
               "这只是本样本的观测，不把高聚类系数解释为流派分类准确。", "",
               "## E5：谱和强节点域", "",
               "| k | 分量内部 λ₂ | 选中向量 | 节点域规模 | normalized cut | 最大特征对残差 |",
               "|---:|---:|---:|---|---:|---:|"]
    for s in summaries:
        k, e = s["k"], s["e5"]
        sizes = ", ".join(str(d["nodes"]) for d in e["domains"])
        report.append(f"| {k} | {e['lambda2']:.6f} | q{e['selected_index_1_based']} | {sizes} | {e['normalized_cut']:.6f} | {e['max_eigenpair_residual']:.2e} |")
    report += ["", "三张图的 q₂ 都满足预定的非零、双侧谱隙和数值精度条件；正负诱导子图各自连通，均得到两个强节点域。",
               "k=2 的 0.018170 是最大分量内部 λ₂；k=2 全图不连通，其全图 λ₂=0。",
               "k=3..64 全图 λ₂ 曲线见 [fiedler_vs_k.png](fiedler_vs_k.png)。",
               "节点域编号与特征向量符号任意，跨 k 比较依据全局 node_index 的成员对应。", "",
               "## 节点域成员稳定性", "",
               "先按各域平均 onset rate 对齐为低／高组，再在共有歌曲上比较。ARI=1 表示两次划分完全一致。",
               "", "| k 对 | 共有歌曲 | 对齐后同组比例 | ARI |", "|---|---:|---:|---:|"]
    for ka, kb, n, agreement, ari, *_ in comparisons:
        report.append(f"| {ka} vs {kb} | {n} | {agreement:.2%} | {ari:.4f} |")
    report += ["", "## 原始数据解释", "",
               "各 k 的 raw_feature_association.csv 给出域间均值的 η²；genre_enrichment.csv 和 mood_theme_enrichment.csv 仅以该标签族有标注的歌曲作分母。",
               "域内艺术家计数与最大占比见 domain_artist_summary.json。声学特征参与建图，因此 η² 是描述性关联，不是独立预测成绩。",
               "", "## 文件入口", "",
               "各 kXXX 目录包含 e4_distributions.png、e5_spectrum.png、e5_nodal_domains.png、e5_raw_features.png、两类标签富集热图及前三个高频标签的三维着色图。",
               "同目录还保存逐节点统计、33 个特征对、全局节点索引和原始数值表。", ""]
    (out / "COMPARISON.md").write_text("\n".join(report), encoding="utf-8")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    out = run / "selected_k"
    summaries = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    curve = fiedler_sweep(run, out)
    for s in summaries:
        if s["k"] >= 3:
            assert abs(s["e5"]["lambda2"] - curve[s["k"] - 2][2]) < 1e-8
    comparisons = compare_partitions(out)
    report(out, summaries, comparisons)
    print(f"Wrote {out / 'COMPARISON.md'}", flush=True)

if __name__ == "__main__":
    main()

