"""Full E1-E3 using reference feature/PCA/graph definitions and bounded memory."""
import argparse
import csv
import gzip
import hashlib
import html
import json
import platform
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
from numpy.lib.format import open_memmap
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from download import digest
import reference_build_graph as ref

ROOT = Path(__file__).resolve().parents[1]
BLOCK = 256


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def metadata(name):
    with (ROOT / 'source_metadata' / name).open(encoding='utf-8', newline='') as stream:
        rows = list(csv.reader(stream, delimiter='\t'))[1:]
    return {r[3].replace('.mp3', '.json'): r for r in rows}


def prepare():
    target = ROOT / 'data'
    target.mkdir(exist_ok=True)
    if (target / 'audit.json').exists():
        audit = json.loads((target / 'audit.json').read_text(encoding='utf-8'))
        for item in audit['outputs']:
            if digest(target / item['file']) != item['sha256']:
                raise ValueError('Prepared input changed; choose a new directory')
        return
    state = json.loads((ROOT / 'source_metadata/download_manifest.json').read_text(encoding='utf-8'))
    if not state.get('complete'):
        raise ValueError('Download must complete before data preparation')
    checks = dict((line.split()[1], line.split()[0]) for line in (ROOT / 'source_metadata/raw_30s_acousticbrainz_sha256_tracks.txt').read_text().splitlines())
    mood_checks = dict((line.split()[1], line.split()[0]) for line in (ROOT / 'source_metadata/autotagging_moodtheme_acousticbrainz_sha256_tracks.txt').read_text().splitlines())
    full, mood, rawmeta = metadata('raw_30s_cleantags_50artists.tsv'), metadata('autotagging_moodtheme.tsv'), metadata('raw_30s.tsv')
    if not set(mood).issubset(full):
        raise ValueError('Mood metadata is not a subset of the full tagging set')
    records, seen, exclusions, clipped = [], set(), [], 0
    for archive in state['archives']:
        path = ROOT / 'archives' / archive['file']
        if digest(path) != archive['sha256']:
            raise ValueError(f'Archive changed: {path.name}')
        with tarfile.open(path, 'r:gz') as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith('.json'):
                    continue
                if member.name in seen or member.name not in checks:
                    raise ValueError(f'Unexpected or duplicate member: {member.name}')
                seen.add(member.name)
                original = tar.extractfile(member).read()
                sha = hashlib.sha256(original).hexdigest()
                if sha != checks[member.name]:
                    raise ValueError(f'Track checksum mismatch: {member.name}')
                if member.name in mood_checks and sha != mood_checks[member.name]:
                    raise ValueError(f'Mood/full feature mismatch: {member.name}')
                if member.name not in full:
                    continue
                row = full[member.name]
                j = json.loads(original)
                try:
                    features, count = ref.extract_features(j)
                except (KeyError, ValueError, TypeError) as error:
                    exclusions.append({'track_id': row[0], 'member': member.name, 'reason': str(error), 'in_moodtheme': member.name in mood})
                    continue
                clipped += count
                node = dict(track_id=row[0], artist_id=row[1], album_id=row[2],
                            audio_path_in_full_dataset=row[3], duration_seconds=float(row[4]),
                            source_archive=archive['file'], source_member=member.name, original_sha256=sha,
                            in_moodtheme=member.name in mood,
                            **{category: [tag.split('---', 1)[1] for tag in row[5:] if tag.startswith(category + '---')]
                               for category in ['genre', 'instrument', 'mood/theme']})
                acoustic = {'lowlevel': {'mfcc': j['lowlevel']['mfcc']},
                            'tonal': {'hpcp': j['tonal']['hpcp']},
                            'rhythm': {'bpm': j['rhythm']['bpm'], 'onset_rate': j['rhythm']['onset_rate']}}
                records.append((node, features, acoustic))
        print(f'Extracted {archive["file"]}: {len(seen)} verified JSONs, {len(records)} eligible tracks', flush=True)
    if seen != set(checks):
        raise ValueError(f'Archive coverage mismatch: {len(set(checks) - seen)} missing tracks')
    missing = [{'track_id': row[0], 'member': member, 'reason': 'no official acoustic feature JSON', 'in_moodtheme': member in mood}
               for member, row in full.items() if member not in seen]
    exclusions.extend(missing)
    records.sort(key=lambda record: record[0]['track_id'])
    if len({r[0]['track_id'] for r in records}) != len(records):
        raise ValueError('Duplicate track IDs')
    with gzip.open(target / 'tracks.jsonl.gz', 'wt', encoding='utf-8') as stream:
        for node, features, acoustic in records:
            stream.write(json.dumps({'node': node, 'acoustic': acoustic}, ensure_ascii=False) + '\n')
    np.save(target / 'features_raw.npy', np.asarray([r[1] for r in records]))
    with (target / 'nodes.jsonl').open('w', encoding='utf-8') as stream:
        for index, (node, _, _) in enumerate(records):
            stream.write(json.dumps(dict(node, node_index=index), ensure_ascii=False) + '\n')
    write_json(target / 'exclusions.json', exclusions)
    audit = dict(raw_30s_metadata_tracks=len(rawmeta), official_feature_tracks=len(checks), verified_feature_tracks=len(seen),
                 full_metadata_tracks=len(full), moodtheme_metadata_tracks=len(mood),
                 eligible_full_tracks=len(records), eligible_moodtheme_tracks=sum(r[0]['in_moodtheme'] for r in records),
                 excluded_tracks=len(exclusions), clipped_variances=clipped,
                 mood_features_match_full_archives=True,
                 outputs=[{'file': name, 'sha256': digest(target / name)} for name in ['tracks.jsonl.gz', 'features_raw.npy', 'nodes.jsonl', 'exclusions.json']])
    write_json(target / 'audit.json', audit)
    print('DATA AUDIT:', json.dumps(audit), flush=True)


def exact_neighbors(distances, k=64):
    """Exact top-k, deterministic distance/index ties, excluding self even for duplicates."""
    n = len(distances)
    indices = np.empty((n, k), dtype=np.int32)
    values = np.empty((n, k), dtype=np.float64)
    for start in range(0, n, BLOCK):
        end = min(n, start + BLOCK)
        block = np.array(distances[start:end], copy=True)
        block[np.arange(end-start), np.arange(start, end)] = np.inf
        candidates = np.argpartition(block, k-1, axis=1)[:, :k]
        for local, cand in enumerate(candidates):
            row = block[local]
            threshold = row[cand].max()
            # Include all boundary ties, then use node index as the secondary key.
            chosen = np.flatnonzero(row <= threshold)
            chosen = chosen[np.lexsort((chosen, row[chosen]))][:k]
            indices[start+local] = chosen
            values[start+local] = row[chosen]
    return indices, values


def create_distances(features, path):
    n = len(features)
    temporary = path.with_suffix('.partial.npy')
    output = open_memmap(temporary, mode='w+', dtype=np.float64, shape=(n, n))
    for start in range(0, n, BLOCK):
        end = min(n, start + BLOCK)
        for right in range(start, n, BLOCK):
            stop = min(n, right + BLOCK)
            block = cdist(features[start:end], features[right:stop], metric='euclidean')
            if start == right:
                block = np.triu(block) + np.triu(block, 1).T
            output[start:end, right:stop] = block
            if start != right:
                output[right:stop, start:end] = block.T
        if start // BLOCK % 20 == 0:
            print(f'E2 distances: {end}/{n} rows', flush=True)
    output.flush()
    del output
    temporary.replace(path)


def create_similarities(distances, sigma, path):
    n = len(distances)
    temporary = path.with_suffix('.partial.npy')
    output = open_memmap(temporary, mode='w+', dtype=np.float64, shape=(n, n))
    bins = np.linspace(0, 1, 101)
    hist = np.zeros(100, dtype=np.int64)
    total, zeros, duplicate_pairs = 0., 0, 0
    for start in range(0, n, BLOCK):
        end = min(n, start + BLOCK)
        for right in range(start, n, BLOCK):
            stop = min(n, right + BLOCK)
            d = np.asarray(distances[start:end, right:stop])
            w = np.exp(-0.5 * (d / sigma)**2)
            if start == right:
                np.fill_diagonal(w, 0)
                mask = np.triu_indices(len(w), 1)
                v, dv = w[mask], d[mask]
            else:
                v, dv = w.ravel(), d.ravel()
            hist += np.histogram(v, bins=bins)[0]
            total += float(v.sum())
            zeros += int(np.count_nonzero(v == 0))
            duplicate_pairs += int(np.count_nonzero(dv == 0))
            output[start:end, right:stop] = w
            if start != right:
                output[right:stop, start:end] = w.T
        if start // BLOCK % 20 == 0:
            print(f'E2 similarities: {end}/{n} rows', flush=True)
    output.flush()
    del output
    temporary.replace(path)
    return dict(unordered_pairs=n*(n-1)//2, mean_offdiagonal_similarity=total/(n*(n-1)//2),
                zero_similarity_pairs=zeros, zero_distance_pairs=duplicate_pairs,
                similarity_histogram={'bin_edges': bins.tolist(), 'counts': hist.tolist()})


def canonical_components(graph):
    count, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    minimum = np.full(count, len(labels), dtype=np.int64)
    np.minimum.at(minimum, labels, np.arange(len(labels)))
    order = np.lexsort((minimum, -sizes))
    lookup = np.empty(count, dtype=np.int32)
    lookup[order] = np.arange(count)
    return lookup[labels], sizes[order]


def run(name):
    started = time.time()
    out = ROOT / 'experiments' / name
    out.mkdir(exist_ok=True)
    state = json.loads((ROOT / 'source_metadata/download_manifest.json').read_text(encoding='utf-8'))
    audit = json.loads((ROOT / 'data/audit.json').read_text(encoding='utf-8'))
    config = dict(experiment=name, scope='complete mood/theme subset' if name == 'moodtheme' else 'complete cleaned 195-tag autotagging set',
                  dimensions=32, seed=521, k_min=2, k_max=64, precision='float64',
                  neighbor_method='exact blocked Euclidean distance; distance then node_index ties',
                  block_rows=BLOCK, symmetrization='union', bandwidth_neighbor=10,
                  upstream=state['upstream'], reference_pipeline=state['pipeline'],
                  reference_code_sha256=digest(ROOT / 'code/reference_build_graph.py'),
                  implementation_sha256=digest(Path(__file__)), input_audit_sha256=digest(ROOT / 'data/audit.json'),
                  python=sys.version, numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, platform=platform.platform())
    config_path = out / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != config:
        raise ValueError(f'Config/implementation changed: {name}; use a new experiment directory')
    write_json(config_path, config)
    if (out / 'COMPLETE.json').exists():
        print(f'{name}: already complete; use validate.py for artifact checks', flush=True)
        return
    if not (out / 'E1.json').exists():
        all_nodes = [json.loads(line) for line in (ROOT / 'data/nodes.jsonl').read_text(encoding='utf-8').splitlines()]
        mask = np.asarray([node['in_moodtheme'] or name == 'full' for node in all_nodes])
        nodes = [dict(node, node_index=i) for i, node in enumerate(node for node, use in zip(all_nodes, mask) if use)]
        raw = np.load(ROOT / 'data/features_raw.npy')[mask]
        with threadpool_limits(limits=4):
            features, params = ref.preprocess(raw, 32)
        np.save(out / 'features_raw.npy', raw)
        np.save(out / 'features.npy', features)
        np.savez_compressed(out / 'preprocessing.npz', **params)
        write_json(out / 'feature_names.json', ref.FEATURE_NAMES)
        with (out / 'nodes.jsonl').open('w', encoding='utf-8') as stream:
            for node in nodes:
                stream.write(json.dumps(node, ensure_ascii=False) + '\n')
        tag_counts = Counter(tag for node in nodes for category in ['genre', 'instrument', 'mood/theme'] for tag in [category+'---'+t for t in node[category]])
        write_json(out / 'E1.json', dict(n_nodes=len(nodes), n_artists=len({node['artist_id'] for node in nodes}),
                   raw_dimensions=100, pca_dimensions=features.shape[1],
                   cumulative_explained_variance=float(params['explained_variance_ratio'].sum()),
                   original_metadata_tracks=audit['moodtheme_metadata_tracks' if name == 'moodtheme' else 'full_metadata_tracks'],
                   excluded_tracks=(audit['moodtheme_metadata_tracks'] if name == 'moodtheme' else audit['full_metadata_tracks'])-len(nodes),
                   constant_feature_columns=np.flatnonzero(~params['keep_mask']).tolist(),
                   tag_counts=dict(tag_counts), category_coverage={c: sum(bool(node[c]) for node in nodes) for c in ['genre', 'instrument', 'mood/theme']}))
        print(f'{name} E1 COMPLETE: {len(nodes)} tracks, PCA variance {params["explained_variance_ratio"].sum():.6%}', flush=True)
    features = np.load(out / 'features.npy')
    if not (out / 'E2.json').exists():
        path = out / 'distances.npy'
        if not path.exists():
            create_distances(features, path)
        distances = np.load(path, mmap_mode='r')
        neighbors, neighbor_distances = exact_neighbors(distances)
        kth = neighbor_distances[:, 9]
        sigma = float(np.median(kth))
        fallback = None
        if sigma == 0:
            positive = kth[kth > 0]
            fallback = 'median positive 10th-neighbor distance'
            if not positive.size:
                raise ValueError('All 10th-neighbor distances zero; explicit pairwise fallback required')
            sigma = float(np.median(positive))
        weights = np.exp(-0.5 * (neighbor_distances / sigma)**2)
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError('Nonfinite or underflowed retained neighbor weights')
        np.savez_compressed(out / 'analysis_data.npz', features=features, neighbors=neighbors, neighbor_weights=weights)
        np.save(out / 'neighbor_distances.npy', neighbor_distances)
        stats = create_similarities(distances, sigma, out / 'similarities.npy')
        write_json(out / 'E2.json', dict(sigma=sigma, bandwidth_neighbor=10, bandwidth_fallback=fallback,
                   matrix_shape=list(distances.shape), matrix_dtype='float64', similarity_diagonal=0,
                   formula='W_ij=exp(-R_ij^2/(2*sigma^2)) for i!=j; W_ii=0', **stats))
        del distances
        print(f'{name} E2 COMPLETE: sigma={sigma:.9f}', flush=True)
    if not (out / 'E3.json').exists():
        with np.load(out / 'analysis_data.npz') as bundle:
            neighbors, weights = bundle['neighbors'], bundle['neighbor_weights']
        (out / 'graphs').mkdir(exist_ok=True)
        (out / 'components').mkdir(exist_ok=True)
        rows = []
        for k in range(2, 65):
            graph = ref.build_graph(neighbors, weights, k)
            labels, sizes = canonical_components(graph)
            sparse.save_npz(out / f'graphs/k{k:03d}.npz', graph)
            np.savez_compressed(out / f'components/k{k:03d}.npz', labels=labels, sizes=sizes)
            rows.append(dict(k=k, n_nodes=len(features), n_edges=graph.nnz//2,
                             n_components=len(sizes), largest_component_size=int(sizes[0]),
                             largest_component_fraction=float(sizes[0]/len(features)), component_sizes=sizes.tolist()))
            print(f'{name} E3 k={k}: edges={graph.nnz//2}, components={len(sizes)}, largest={sizes[0]}', flush=True)
        write_json(out / 'connectivity.json', rows)
        with (out / 'connectivity.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=[key for key in rows[0] if key != 'component_sizes'])
            writer.writeheader()
            writer.writerows({key: value for key, value in row.items() if key != 'component_sizes'} for row in rows)
        write_json(out / 'E3.json', dict(k_min=2, k_max=64, graph_count=63,
                   k_c=next((r['k'] for r in rows if r['n_components']==1), None),
                   k_c_scope='minimum connected k within scanned range 2..64 (k=1 not tested)',
                   k2=rows[0], k64=rows[-1], nested_edge_sets=True))
    render(name)
    write_json(out / 'COMPLETE.json', dict(experiment=name, phases=['E1', 'E2', 'E3'],
                                         elapsed_seconds=time.time()-started, independently_validated=False))


def render(name):
    out = ROOT / 'experiments' / name
    e1, e2, e3 = [json.loads((out / f'E{i}.json').read_text(encoding='utf-8')) for i in range(1, 4)]
    rows = json.loads((out / 'connectivity.json').read_text(encoding='utf-8'))
    with np.load(out / 'preprocessing.npz') as params:
        ratios = params['explained_variance_ratio']
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(ratios)+1), np.cumsum(ratios), marker='.')
    ax.set(xlabel='Retained principal components', ylabel='Cumulative explained variance', title=f'{name}: E1 PCA (group-balanced features)')
    ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out / 'E1_pca.png', dpi=160); plt.close(fig)
    hist = e2['similarity_histogram']
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(np.asarray(hist['bin_edges'][:-1]), hist['counts'], width=.01, align='edge')
    ax.set(xlabel='Gaussian similarity (off-diagonal)', ylabel='Unordered pair count', title=f'{name}: E2 all-pair similarities')
    fig.tight_layout(); fig.savefig(out / 'E2_similarity.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot([r['k'] for r in rows], [r['n_components'] for r in rows], marker='.')
    axes[0].set(ylabel='Connected components', title=f'{name}: E3 union-kNN connectivity')
    axes[1].plot([r['k'] for r in rows], [r['largest_component_fraction'] for r in rows], marker='.')
    axes[1].set(xlabel='k', ylabel='Largest component / n')
    for ax in axes: ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / 'E3_connectivity.png', dpi=160); plt.close(fig)
    text = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>{name} E1–E3</title>
<style>body{{font:16px/1.7 system-ui;max-width:1000px;margin:40px auto;padding:0 20px}}img{{max-width:100%}}pre{{white-space:pre-wrap;background:#f2f4f8;padding:20px}}</style>
<h1>{name} 完整数据集 E1–E3</h1>
<p>官方 Essentia/AcousticBrainz 预计算统计。无抽样；歌曲标签不参与特征、PCA、距离、带宽或建图。</p>
<h2>E1 特征与 PCA</h2><p>标签表 {e1['original_metadata_tracks']:,} 首；有效节点 {e1['n_nodes']:,}；排除 {e1['excluded_tracks']} 首；艺术家 {e1['n_artists']:,}。
原始 100 维（MFCC mean/std 26、HPCP mean/std 72、BPM/onset rate 2）；总体标准差 z-score，各组按有效维数平方根缩放，full-SVD PCA 32 维、不白化；累计解释方差 {e1['cumulative_explained_variance']:.4%}。</p><img src="E1_pca.png">
<h2>E2 完整交互矩阵</h2><p>完整 float64 欧氏距离与 Gaussian 相似度矩阵，维度 {e1['n_nodes']} × {e1['n_nodes']}。
σ={e2['sigma']:.9f}，取每个节点第 10 个非自身近邻距离的中位数。W=exp(-R²/(2σ²))，相似度对角线为 0。
不同 ID 的零距离歌曲对 {e2['zero_distance_pairs']}，仍保留独立节点；数值下溢为零的非邻居相似度对 {e2['zero_similarity_pairs']}，不影响已核验为正的保留图边。</p><img src="E2_similarity.png">
<h2>E3 kNN 图与连通性</h2><p>精确近邻，距离相同按 node_index 排序，排除自身；并集对称化，边权采用 E2 的相似度。扫描 k=2…64，固定邻居序列和 σ，共 63 张 CSR 图。
k=2 时 {e3['k2']['n_components']} 个分量，最大分量 {e3['k2']['largest_component_size']:,} 个节点。
扫描范围内首个连通 k：{e3['k_c']}。未测试 k=1，不声称是所有正整数 k 的全局阈值。</p><img src="E3_connectivity.png">
<p>nodes.jsonl 的 node_index 对齐所有矩阵与图；components/kXXX.npz 的 labels 按分量规模降序重编号，0 为最大分量。
完整方法/软件/来源见 config.json，排除原因见 ../../data/exclusions.json；独立验证结果见 validation.json。</p>
<h2>使用限制</h2><p>两实验在不同总体上独立拟合 PCA 和 σ，故边权不直接跨实验比较。标签缺失不代表属性不存在；同艺术家及重复声学特征可造成相关性。这里未运行 E4–E7。</p></html>'''
    (out / 'report.html').write_text(text, encoding='utf-8')


def comparison():
    records = []
    for name in ['moodtheme', 'full']:
        out = ROOT / 'experiments' / name
        a, b, c = [json.loads((out / f'E{i}.json').read_text(encoding='utf-8')) for i in range(1, 4)]
        records.append(dict(experiment=name, n_nodes=a['n_nodes'], n_artists=a['n_artists'],
                            excluded_tracks=a['excluded_tracks'], pca_explained_variance=a['cumulative_explained_variance'],
                            sigma=b['sigma'], k2_edges=c['k2']['n_edges'], k2_components=c['k2']['n_components'],
                            k2_largest_fraction=c['k2']['largest_component_fraction'], first_connected_k_in_2_64=c['k_c']))
    write_json(ROOT / 'comparison.json', records)
    with (ROOT / 'comparison.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    table = '<tr><th>指标</th><th>mood/theme</th><th>full</th></tr>' + ''.join('<tr><td>'+html.escape(key)+'</td>'+''.join('<td>'+html.escape(str(r[key]))+'</td>' for r in records)+'</tr>' for key in records[0])
    (ROOT / 'comparison.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.6 system-ui;margin:40px}td,th{padding:10px;border:1px solid #ddd}table{border-collapse:collapse}</style><h1>完整 MTG-Jamendo：两实验 E1–E3 对比</h1><table>'+table+'</table><p>两个实验无抽样，各自拟合标准化、PCA 和带宽。full 包含 mood/theme 的有效节点，但 PCA 空间和近邻图分别重建。</p><p><a href="experiments/moodtheme/report.html">mood/theme 实验报告</a> · <a href="experiments/full/report.html">全量实验报告</a></p>', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=['moodtheme', 'full', 'both'], default='both')
    args = parser.parse_args()
    prepare()
    for name in (['moodtheme', 'full'] if args.experiment == 'both' else [args.experiment]):
        run(name)
    if all((ROOT / 'experiments' / name / 'COMPLETE.json').exists() for name in ['moodtheme', 'full']):
        comparison()
