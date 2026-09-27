"""E4/E5 analysis for selected saved Jamendo graphs, with a k=2 baseline.

Run from any directory: python code/analyze_selected_k.py --run-dir results
Inputs are read only. Outputs live in RUN/selected_k.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.sparse.linalg import eigsh

SEED = 521
KS = (2, 3, 64)
COLORS = ("#2869aa", "#d87532", "#43916c", "#8f66aa")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def save_fig(fig, path):
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def graph_at(run, k):
    path = run / "graphs" / f"k{k:03d}.npz"
    a = sparse.load_npz(path).tocsr()
    a.eliminate_zeros()
    assert a.shape == (2000, 2000)
    assert (a != a.T).nnz == 0 and np.all(a.diagonal() == 0)
    assert np.isfinite(a.data).all() and np.all(a.data > 0)
    return a, hashlib.sha256(path.read_bytes()).hexdigest()


def components(a):
    count, membership = connected_components(a, directed=False)
    groups = [np.flatnonzero(membership == c) for c in range(count)]
    return sorted(groups, key=lambda ids: (-len(ids), int(ids[0])))


def e4(a, ids, out, k):
    binary = a.copy()
    binary.data = np.ones_like(binary.data)
    degree = np.diff(binary.indptr)
    twice_triangles = np.asarray((binary @ binary).multiply(binary).sum(axis=1)).ravel()
    triangles = (twice_triangles / 2).astype(int)
    possible_twice = degree * (degree - 1)
    clustering = np.divide(twice_triangles, possible_twice,
                           out=np.zeros(len(ids)), where=possible_twice > 0)
    strength = np.asarray(a.sum(axis=1)).ravel()
    assert degree.sum() == a.nnz and np.all((clustering >= 0) & (clustering <= 1))
    assert int(twice_triangles.sum()) % 6 == 0
    write_csv(out / "node_metrics.csv",
              ["global_node_index", "degree", "strength", "triangles", "clustering"],
              zip(ids, degree, strength, triangles, clustering))
    values, counts = np.unique(degree, return_counts=True)
    write_csv(out / "degree_distribution.csv", ["degree", "count", "fraction"],
              zip(values, counts, counts / len(ids)))
    bins = np.linspace(0, 1, 21)
    hist, _ = np.histogram(clustering, bins=bins)
    write_csv(out / "clustering_distribution.csv", ["bin_left", "bin_right", "count", "fraction"],
              zip(bins[:-1], bins[1:], hist, hist / len(ids)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].bar(values, counts / len(ids), color=COLORS[0], width=.85)
    axes[0].set(xlabel="Degree (number of neighbors)", ylabel="Fraction of nodes")
    axes[1].hist(clustering, bins=bins, weights=np.ones(len(ids)) / len(ids), color=COLORS[1])
    axes[1].set(xlabel="Local clustering coefficient", ylabel="Fraction of nodes", xlim=(0, 1))
    fig.suptitle(f"E4 | k={k} | n={len(ids)} | PCA=32 | component 0")
    save_fig(fig, out / "e4_distributions.png")
    return dict(nodes=len(ids), edges=a.nnz // 2, degree_mean=float(degree.mean()),
                degree_median=float(np.median(degree)), degree_min=int(degree.min()),
                degree_max=int(degree.max()), degree_mode=int(values[np.argmax(counts)]),
                clustering_mean=float(clustering.mean()),
                clustering_median=float(np.median(clustering)),
                clustering_zero_fraction=float(np.mean(clustering == 0)),
                clustering_one_fraction=float(np.mean(clustering == 1)),
                triangles=int(twice_triangles.sum() // 6),
                transitivity=float(twice_triangles.sum() / possible_twice.sum()))


def solve(a, number=33):
    l = laplacian(a, normed=True).tocsr()
    n = a.shape[0]
    if number >= n:
        values, vectors = np.linalg.eigh(l.toarray())
    else:
        values, vectors = eigsh(l, k=number, sigma=-1e-6, which="LM", tol=1e-10,
                                v0=np.random.default_rng(SEED).normal(size=n))
        order = np.argsort(values)
        values, vectors = values[order], vectors[:, order]
    residuals = np.linalg.norm(l @ vectors - vectors * values, axis=0)
    orth_error = float(np.max(np.abs(vectors.T @ vectors - np.eye(len(values)))))
    strength = np.asarray(a.sum(axis=1)).ravel()
    null = np.sqrt(strength) / np.linalg.norm(np.sqrt(strength))
    assert strength.min() > 0 and np.linalg.norm(l @ null) < 1e-8
    assert np.max(residuals) <= 1e-7 and orth_error <= 1e-6
    assert abs(values[0]) < 1e-8 and values[1] > 1e-8
    assert values.min() >= -1e-8 and values.max() <= 2 + 1e-8
    for j in range(vectors.shape[1]):
        pivot = np.argmax(np.abs(vectors[:, j]))
        if vectors[pivot, j] < 0:
            vectors[:, j] *= -1
    return l, values, vectors, residuals, orth_error


def nodal_domains(binary, q, ids):
    epsilon = 1e-8 * np.max(np.abs(q))
    sign = np.where(q > epsilon, 1, np.where(q < -epsilon, -1, 0))
    found = []
    for side in (-1, 1):
        local_ids = np.flatnonzero(sign == side)
        if len(local_ids):
            count, labels = connected_components(binary[local_ids][:, local_ids], directed=False)
            for c in range(count):
                members = local_ids[labels == c]
                found.append((members, side))
    found.sort(key=lambda row: (-len(row[0]), int(ids[row[0]].min())))
    domains = np.full(len(q), -1, dtype=int)
    rows = []
    for domain_id, (members, side) in enumerate(found, 1):
        domains[members] = domain_id
        rows.append(dict(domain_id=domain_id, sign=side, nodes=len(members),
                         smallest_global_index=int(ids[members].min())))
    assert np.count_nonzero(domains == -1) == np.count_nonzero(sign == 0)
    return domains, sign, rows, float(epsilon)


def spectral(a, ids, nodes, raw, names, out, k):
    l, values, vectors, residuals, orth = solve(a)
    sparse.save_npz(out / "normalized_laplacian.npz", l)
    left = values[1:32] - values[:31]
    right = values[2:33] - values[1:32]
    isolation = np.minimum(left, right)
    relative = isolation / np.maximum(np.abs(values[1:32]), 1e-6)
    floor = max(1e-8, 10 * float(residuals.max()))
    eligible = (values[1:32] > floor) & (left > floor) & (right > floor) & (isolation >= 1e-3) & (relative >= .05)
    if eligible[0]:
        selected = 1
        selection = "qualified_q2"
    elif eligible.any():
        selected = int(np.argmax(np.where(eligible, isolation, -np.inf))) + 1
        selection = "qualified_max_isolation"
    else:
        selected = None
        selection = "no_well_separated_candidate"
    write_csv(out / "eigenvalues.csv", ["index_1_based", "eigenvalue", "residual", "plotted"],
              ((j + 1, values[j], residuals[j], j < 32) for j in range(33)))
    write_csv(out / "eigenvalue_candidates.csv",
              ["index_1_based", "eigenvalue", "left_gap", "right_gap", "isolation", "relative_isolation", "qualified"],
              ((j + 2, values[j + 1], left[j], right[j], isolation[j], relative[j], eligible[j]) for j in range(31)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].plot(np.arange(1, 33), values[:32], "o-", ms=3)
    axes[0].set(xlabel="Eigenvalue index (1-based)", ylabel="Eigenvalue")
    axes[1].bar(np.arange(2, 33), isolation, color=COLORS[0])
    axes[1].set(xlabel="Candidate index (1-based)", ylabel="Minimum adjacent gap")
    if selected is not None:
        axes[0].scatter([selected + 1], [values[selected]], color=COLORS[1], zorder=3)
        axes[1].bar([selected + 1], [isolation[selected - 1]], color=COLORS[1])
    fig.suptitle(f"E5 | k={k} | n={len(ids)} | weighted normalized Laplacian")
    save_fig(fig, out / "e5_spectrum.png")
    summary = dict(lambda2=float(values[1]), eigenvalue_count=33,
                   max_eigenpair_residual=float(residuals.max()), orthogonality_error=orth,
                   selection_status=selection,
                   selected_index_1_based=None if selected is None else selected + 1,
                   selected_eigenvalue=None if selected is None else float(values[selected]),
                   selected_isolation=None if selected is None else float(isolation[selected - 1]),
                   selected_relative_isolation=None if selected is None else float(relative[selected - 1]))
    if selected is None:
        write_json(out / "spectral_summary.json", summary)
        return summary
    binary = a.copy()
    binary.data = np.ones_like(binary.data)
    domain_ids, sign, domains, epsilon = nodal_domains(binary, vectors[:, selected], ids)
    coordinates_idx = [selected] + [j for j in (1, 2, 3) if j != selected][:2]
    coordinates = vectors[:, coordinates_idx]
    np.savez_compressed(out / "spectral_data.npz", global_node_index=ids,
                        eigenvalues=values, eigenvectors=vectors, residuals=residuals,
                        embedding=coordinates, embedding_indices=np.array(coordinates_idx) + 1,
                        domain_id=domain_ids, signs=sign, boundary_mask=domain_ids == -1)
    write_csv(out / "node_embedding.csv",
              ["global_node_index", "track_id", "artist_id", "q_x", "q_y", "q_z", "domain_id", "sign"],
              ((int(g), nodes[g]["track_id"], nodes[g]["artist_id"], *coordinates[i],
                int(domain_ids[i]), int(sign[i])) for i, g in enumerate(ids)))
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for row in domains:
        mask = domain_ids == row["domain_id"]
        ax.scatter(*coordinates[mask].T, s=6, alpha=.65, depthshade=False,
                   color=COLORS[(row["domain_id"] - 1) % len(COLORS)],
                   label=f"Domain {row['domain_id']} (n={row['nodes']})")
    if np.any(domain_ids == -1):
        ax.scatter(*coordinates[domain_ids == -1].T, s=8, color="gray", label="boundary")
    ax.set(xlabel=f"q{coordinates_idx[0]+1}", ylabel=f"q{coordinates_idx[1]+1}",
           zlabel=f"q{coordinates_idx[2]+1}")
    ax.set_title(f"k={k} | n={len(ids)} | PCA=32 | domains of q{selected+1}")
    ax.legend(fontsize=8)
    save_fig(fig, out / "e5_nodal_domains.png")
    strength = np.asarray(a.sum(axis=1)).ravel()
    upper = sparse.triu(a, k=1).tocoo()
    crossing = domain_ids[upper.row] != domain_ids[upper.col]
    cut_weight = float(upper.data[crossing].sum())
    summary.update(domain_count=len(domains), boundary_nodes=int(np.sum(domain_ids == -1)),
                   boundary_epsilon=epsilon, domains=domains,
                   crossing_edges=int(crossing.sum()), crossing_weight=cut_weight)
    if len(domains) == 2 and summary["boundary_nodes"] == 0:
        summary["normalized_cut"] = float(sum(cut_weight / strength[domain_ids == r["domain_id"]].sum() for r in domains))
    interpret(ids, nodes, raw, names, domain_ids, domains, coordinates, out, k)
    write_json(out / "spectral_summary.json", summary)
    return summary


def interpret(ids, nodes, raw, names, domain_ids, domains, coordinates, out, k):
    x = raw[ids]
    valid = domain_ids > 0
    x = x[valid]
    labels = domain_ids[valid]
    valid_ids = ids[valid]
    overall_mean = x.mean(axis=0)
    total = ((x - overall_mean) ** 2).sum(axis=0)
    between = np.zeros(x.shape[1])
    feature_rows = []
    domain_rows = []
    for row in domains:
        domain = row["domain_id"]
        mask = labels == domain
        mean = x[mask].mean(axis=0)
        between += mask.sum() * (mean - overall_mean) ** 2
        feature_rows.extend((domain, name, float(mean[j])) for j, name in enumerate(names))
        artists = [nodes[g]["artist_id"] for g in valid_ids[mask]]
        counts = sorted(((artists.count(artist), artist) for artist in set(artists)), reverse=True)
        domain_rows.append(dict(domain_id=domain, nodes=int(mask.sum()), artist_count=len(counts),
                                largest_artist_count=counts[0][0], largest_artist_fraction=counts[0][0] / mask.sum(),
                                top_3_artists=[{"artist_id": a, "count": c} for c, a in counts[:3]],
                                mean_bpm=float(mean[names.index("rhythm.bpm")]),
                                mean_onset_rate=float(mean[names.index("rhythm.onset_rate")])))
    eta2 = np.divide(between, total, out=np.zeros_like(between), where=total > 0)
    order = np.argsort(-eta2)
    write_csv(out / "raw_feature_association.csv", ["feature", "eta_squared"],
              ((names[j], eta2[j]) for j in order))
    write_csv(out / "domain_feature_means.csv", ["domain_id", "feature", "raw_mean"], feature_rows)
    write_json(out / "domain_artist_summary.json", domain_rows)
    std = x.std(axis=0)
    z = np.divide(x - overall_mean, std, out=np.zeros_like(x), where=std > 0)
    means = np.array([z[labels == row["domain_id"]].mean(axis=0) for row in domains])
    top = order[:10]
    fig, ax = plt.subplots(figsize=(11, max(3, len(domains) * .5 + 2)), layout="constrained")
    vmax = max(.1, np.max(np.abs(means[:, top])))
    im = ax.imshow(means[:, top], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(top)), [names[j] for j in top], rotation=40, ha="right")
    ax.set_yticks(range(len(domains)), [f"Domain {r['domain_id']}" for r in domains])
    ax.set_title(f"k={k} | raw feature means (z scores within component)")
    fig.colorbar(im, ax=ax)
    save_fig(fig, out / "e5_raw_features.png")
    for family, key in (("genre", "genre"), ("mood_theme", "mood/theme")):
        universe = np.array([bool(nodes[g][key]) for g in valid_ids])
        tags = sorted({tag for g in valid_ids[universe] for tag in nodes[g][key]})
        if not tags:
            continue
        tag_sets = [set(nodes[g][key]) for g in valid_ids]
        global_counts = {tag: sum(tag in tag_sets[i] for i in np.flatnonzero(universe)) for tag in tags}
        ordered = sorted(tags, key=lambda tag: (-global_counts[tag], tag))
        rows = []
        for row in domains:
            domain = row["domain_id"]
            mask = (labels == domain) & universe
            n = int(mask.sum())
            for tag in tags:
                hits = sum(tag in tag_sets[i] for i in np.flatnonzero(mask))
                global_p = global_counts[tag] / universe.sum()
                p = hits / n if n else None
                rows.append((domain, tag, n, hits, p, global_p, p / global_p if p is not None and global_p else None))
        write_csv(out / f"{family}_enrichment.csv",
                  ["domain_id", "tag", "labeled_in_domain", "hits", "domain_fraction", "component_fraction", "lift"], rows)
        shown = [tag for tag in ordered if global_counts[tag] >= 20][:15]
        if shown:
            values = np.array([[next(r[6] for r in rows if r[0] == d["domain_id"] and r[1] == tag)
                                for tag in shown] for d in domains], dtype=float)
            fig, ax = plt.subplots(figsize=(max(8, len(shown) * .65), max(3, len(domains) * .45 + 2)), layout="constrained")
            im = ax.imshow(values, cmap="YlGnBu", aspect="auto")
            ax.set_xticks(range(len(shown)), shown, rotation=45, ha="right")
            ax.set_yticks(range(len(domains)), [f"Domain {d['domain_id']}" for d in domains])
            ax.set_title(f"k={k} | {family} lift vs labeled component baseline")
            fig.colorbar(im, ax=ax, label="Lift")
            save_fig(fig, out / f"e5_{family}_enrichment.png")
        for tag in ordered[:3]:
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111, projection="3d")
            categories = np.array([2 if not nodes[g][key] else int(tag in nodes[g][key]) for g in ids])
            for c, label, color in ((0, "labeled, absent", "#9ca3af"),
                                    (1, f"{tag} present", "#e07a2f"),
                                    (2, "no family labels", "#343a40")):
                mask = categories == c
                if mask.any():
                    ax.scatter(*coordinates[mask].T, s=5, alpha=.5, color=color, label=label)
            ax.set_title(f"k={k} | {family}: {tag}")
            ax.legend(fontsize=7)
            save_fig(fig, out / f"e5_{family}_tag_{ordered.index(tag)+1}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    out = run / "selected_k"
    out.mkdir(parents=True, exist_ok=True)
    nodes = [json.loads(line) for line in (run / "nodes.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [n["node_index"] for n in nodes] == list(range(2000))
    raw = np.load(run / "features_raw.npy", allow_pickle=False)
    names = json.loads((run / "feature_names.json").read_text(encoding="utf-8"))
    assert raw.shape == (2000, 100) and len(names) == 100
    all_summaries = []
    for k in KS:
        print(f"Analyzing k={k}", flush=True)
        a, graph_hash = graph_at(run, k)
        groups = components(a)
        ids = groups[0]
        assert len(ids) >= 100
        local = a[ids][:, ids].tocsr()
        folder = out / f"k{k:03d}"
        folder.mkdir(exist_ok=True)
        np.save(folder / "global_indices.npy", ids)
        structural = e4(local, ids, folder, k)
        spectral_result = spectral(local, ids, nodes, raw, names, folder, k)
        summary = dict(k=k, graph_sha256=graph_hash, component_sizes=[len(g) for g in groups],
                       e4=structural, e5=spectral_result)
        write_json(folder / "summary.json", summary)
        all_summaries.append(summary)
    write_json(out / "comparison.json", all_summaries)
    write_csv(out / "comparison.csv",
              ["k", "nodes", "edges", "mean_degree", "degree_max", "mean_clustering", "zero_clustering_fraction",
               "lambda2", "selected_eigenvector", "domain_count", "boundary_nodes"],
              ((s["k"], s["e4"]["nodes"], s["e4"]["edges"], s["e4"]["degree_mean"],
                s["e4"]["degree_max"], s["e4"]["clustering_mean"], s["e4"]["clustering_zero_fraction"],
                s["e5"]["lambda2"], s["e5"]["selected_index_1_based"],
                s["e5"].get("domain_count"), s["e5"].get("boundary_nodes")) for s in all_summaries))
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
