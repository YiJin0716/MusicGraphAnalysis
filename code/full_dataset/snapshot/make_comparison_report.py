"""Build a readable comparison from completed and independently validated experiments."""
import html
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_experiments import ROOT


def main():
    experiments = {}
    for name in ['moodtheme', 'full']:
        folder = ROOT / 'experiments' / name
        experiments[name] = {f'E{i}': json.loads((folder / f'E{i}.json').read_text(encoding='utf-8')) for i in range(1, 4)}
        experiments[name]['rows'] = json.loads((folder / 'connectivity.json').read_text(encoding='utf-8'))
        experiments[name]['validation'] = json.loads((folder / 'validation.json').read_text(encoding='utf-8'))
        if not experiments[name]['validation']['passed']:
            raise ValueError('Independent validation must pass before final comparison')
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for name, data in experiments.items():
        rows = data['rows']
        axes[0].plot([r['k'] for r in rows], [r['n_components'] for r in rows], marker='.', label=f"{name} (n={data['E1']['n_nodes']:,})")
        axes[1].plot([r['k'] for r in rows], [r['largest_component_fraction'] for r in rows], marker='.', label=name)
    axes[0].set(ylabel='Connected components', title='Complete MTG-Jamendo: E3 connectivity comparison')
    axes[1].set(xlabel='k (union-kNN)', ylabel='Largest component / n')
    for ax in axes:
        ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(ROOT / 'comparison_connectivity.png', dpi=170); plt.close(fig)
    entries = [
        ('标签表曲目数', lambda d: f"{d['E1']['original_metadata_tracks']:,}"),
        ('有效节点数', lambda d: f"{d['E1']['n_nodes']:,}"),
        ('缺失官方特征的曲目数', lambda d: str(d['E1']['excluded_tracks'])),
        ('艺术家数', lambda d: f"{d['E1']['n_artists']:,}"),
        ('原始特征维数 / PCA 维数', lambda d: '100 / 32'),
        ('PCA 累计解释方差', lambda d: f"{d['E1']['cumulative_explained_variance']:.4%}"),
        ('Gaussian 带宽 σ', lambda d: f"{d['E2']['sigma']:.9f}"),
        ('无序歌曲对数', lambda d: f"{d['E2']['unordered_pairs']:,}"),
        ('零距离的不同 ID 歌曲对', lambda d: f"{d['E2']['zero_distance_pairs']:,}"),
        ('k=2 边数', lambda d: f"{d['E3']['k2']['n_edges']:,}"),
        ('k=2 连通分量数', lambda d: str(d['E3']['k2']['n_components'])),
        ('k=2 最大分量节点数', lambda d: f"{d['E3']['k2']['largest_component_size']:,}"),
        ('k=2 最大分量占比', lambda d: f"{d['E3']['k2']['largest_component_fraction']:.4%}"),
        ('k=2…64 首个全图连通 k', lambda d: str(d['E3']['k_c'])),
        ('保存图数', lambda d: str(d['E3']['graph_count'])),
        ('独立验证', lambda d: '通过（全矩阵、精确近邻、63 张图及全部分量）'),
    ]
    table = '<tr><th>指标</th><th>完整 mood/theme</th><th>全量标签集</th></tr>'
    for label, value in entries:
        table += '<tr><td>'+label+'</td>'+''.join('<td>'+html.escape(value(data))+'</td>' for data in experiments.values())+'</tr>'
    config = json.loads((ROOT / 'source_metadata/download_manifest.json').read_text(encoding='utf-8'))
    summary = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>MTG-Jamendo 完整数据集 E1–E3 对比</title>
<style>body{{font:16px/1.7 system-ui;max-width:1080px;margin:40px auto;padding:0 20px;color:#202632}}h1{{font-size:28px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}th{{background:#edf2f7}}img{{max-width:100%}}code{{background:#f1f3f5;padding:2px 5px}}a{{color:#1764a0}}</style>
<h1>MTG-Jamendo：两个完整数据集实验的 E1–E3</h1>
<p>两个实验均已完成并通过独立验证。使用官方 Essentia/AcousticBrainz 预计算声学特征，不抽样。
完整 mood/theme 与全量标签集分别拟合标准化、分组权重、32 维 PCA 和 Gaussian 带宽，独立建立 k=2…64 的无向加权 union-kNN 图。</p>
<table>{table}</table><h2>连通性比较</h2><img src="comparison_connectivity.png">
<p>数据扩大后，原 2,000 首实验的连通阈值 k=3 不能直接沿用。表中阈值来自各自固定预处理与近邻序列下的完整扫描；它反映该实验的图构造，并不单独说明音乐存在多少语义类别。</p>
<h2>数据覆盖与一致性</h2><p>完整下载 100 个 raw_30s 官方特征分包（1,315,864,188 bytes），校验全部 55,699 个 JSON 的 SHA-256。
全量标签表 55,609 首，其中 <code>track_1057640</code> 和 <code>track_1118596</code> 在官方特征清单中缺失，因此有效节点为 55,607。
mood/theme 的 18,486 首均有可用官方特征，其逐曲校验值与 mood/theme 官方清单一致。
排除记录见 <a href="data/exclusions.json">exclusions.json</a>，完整审计见 <a href="data/audit.json">audit.json</a>。</p>
<p>分块方法已在原 2,000 首样本上与已验证 pipeline 对照：精确近邻索引、所有 63 张图及边权一致；额外核验重复特征和边界并列排序。
两个新实验又独立重读全部矩阵条目，核验欧氏距离、对称性、Gaussian 公式、精确近邻、PCA 重构、全部图和逐节点分量标签。</p>
<h2>报告与后续加载</h2><p><a href="experiments/moodtheme/report.html">完整 mood/theme 报告</a> · <a href="experiments/full/report.html">全量标签集报告</a> · <a href="README.txt">目录、重现命令与加载示例</a> · <a href="comparison.csv">对比 CSV</a></p>
<p>每个实验目录保存完整 <code>distances.npy</code> 与 <code>similarities.npy</code>、PCA 特征、<code>graphs/kXXX.npz</code>、<code>components/kXXX.npz</code> 和 <code>nodes.jsonl</code>，后续 E4–E7 可直接读取。
完整矩阵应使用 <code>np.load(path, mmap_mode='r')</code> 读取。</p>
<h2>解释边界</h2><p>全量实验包含 mood/theme 的歌曲，但重新拟合的 PCA 空间与 σ 不同，边权无法直接跨实验比较；全量图的 mood/theme 子图也不等同于独立 mood/theme 图。
标签不参与建图；标签缺失不代表属性不存在。同艺术家和重复声学特征造成的相关性应在后续分析中考虑。本次范围为 E1–E3。</p>
<p>数据来源：<a href="https://github.com/MTG/mtg-jamendo-dataset/tree/{config['upstream']['commit']}">MTG 官方仓库</a>；
参考 pipeline：<a href="https://github.com/YiJin0716/MusicGraphAnalysis/tree/{config['pipeline']['commit']}">MusicGraphAnalysis</a>。
引用与许可保存在 source_metadata/。</p></html>'''
    (ROOT / 'comparison.html').write_text(summary, encoding='utf-8')
    print('Validated comparison report generated', flush=True)


if __name__ == '__main__':
    main()
