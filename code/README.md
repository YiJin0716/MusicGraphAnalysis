# MusicGraphAnalysis 使用说明

MusicGraphAnalysis 对 2000 首歌曲的声学统计数据进行特征提取、PCA 降维和 kNN 建图，用于分析歌曲之间的相似关系与图的连通性。

项目包含三个文件夹：

- `source_data/`：`tracks.jsonl.gz` 是 2000 首歌曲的原始声学统计字段和元数据；另有来源记录 `SOURCE.json`、引用和上游许可说明。
- `code/`：`build_graph.py` 是处理入口，`load_graph.py` 用于读取已有图，`requirements.txt` 是依赖，本文件是使用说明。
- `results/`：提取的 100 维特征、32 维 PCA 特征、距离和相似度矩阵、近邻、63 张图、连通分量、统计表和图片。

## 输入数据与处理流程

`tracks.jsonl.gz` 每行包含 node（歌曲、艺术家、标签等）和 acoustic（MFCC 均值/协方差、HPCP 均值/方差、BPM 和 onset rate）。这些字段未标准化、未加权、未 PCA；不含音频，也不含未使用的其它声学字段。源文件路径仅为溯源信息。

处理流程：统计字段 → 提取 100 维特征 → 逐列标准化 → 分组加权 → PCA → 欧氏距离和高斯相似度 → 并集对称化 kNN 图 → 连通性统计。

## 运行

在 `MusicGraphAnalysis` 根目录执行：

```powershell
python -m pip install -r code/requirements.txt
python code/build_graph.py --save-graphs
python code/load_graph.py --k 32
python code/load_graph.py --k 2 --largest-component
```

`build_graph.py` 默认读取 `source_data/tracks.jsonl.gz`，生成到 `results/rebuilt/`。输出目录必须为空；再次运行时指定新的目录，例如：

```powershell
python code/build_graph.py --output results/run2 --save-graphs
```

默认输入、输出根据脚本位置定位，与终端目录无关；显式传入的相对路径相对于终端目录。

## 新生成结果的格式

`build_graph.py` 在指定输出目录生成以下文件：

- `features_raw.npy`、`feature_names.json`、`nodes.jsonl`：提取后的特征和对齐信息。
- `preprocessing.npz`：标准化及 PCA 参数。
- `analysis_data.npz`：包含 `features`、`neighbors` 和 `neighbor_weights` 三个数组。
- `connectivity.json`、`summary.json`：连通性及本轮参数/统计。
- `graphs/kXXX.npz`：加 `--save-graphs` 时保存，k 默认从 2 到 64。

`results/` 中还包含完整距离矩阵、相似度矩阵、逐节点连通分量文件和图片。这些文件不由上述命令生成。未使用 `--save-graphs` 时，可用脚本的 `build_graph` 函数从近邻及权重还原图。

运行建图命令后，可在项目根目录读取结果：

```python
import numpy as np
from scipy.sparse import load_npz

with np.load('results/rebuilt/analysis_data.npz') as data:
    F = data['features']  # (2000, 32)
A = load_npz('results/rebuilt/graphs/k032.npz')
```

`load_graph.py` 默认加载直接位于 `results/` 的已有完整结果，而不是 `rebuilt/`。

## E4 / E5 分析

在 `MusicGraphAnalysis` 根目录运行：

```powershell
python code/analyze_e4.py
python code/analyze_e5.py
```

E4 使用 k=2 图的无权连接结构，输出度及局部聚类系数分布到 `results/E4/_degreeDistribution/`。
E5 使用同一张图的原始高斯边权，输出归一化拉普拉斯矩阵、32 个最小特征值图、三维谱嵌入、节点域划分及原始数据对比到 `results/E5/`，并计算 k=3…64 的全图 Fiedler 曲线。报告分别为 `E4_report.md` 和 `E5_report.md`，脚本重运行会更新对应结果文件。

## 完整英文 PDF 报告

`MusicGraphAnalysis_Report.pdf` 覆盖 E1–E5 和 E7，包含数据与特征说明、相似度矩阵、kNN 连通性、分布分析、谱分析、节点域与原始数据的关系以及讨论。

在项目根目录执行 `python code/build_report.py` 可从现有数据和 E4/E5 结果生成报告。报告使用标准 LaTeX `article` 样式，正文源文件为 `code/MusicGraphAnalysis_Report.tex`，需要 MiKTeX 或 TeX Live 提供 `pdflatex`。生成脚本会核对输入数据与图结构，图表素材位于 `results/report_assets/`，Pypdf 用于 PDF 内容检查。

## 当前已有结果与验证

`results/features_raw.npy` 是 (2000,100)，`results/features.npy` 是 (2000,32)。`results/analysis_data.npz` 集中保存 PCA 特征、近邻索引和近邻权重。矩阵顺序与 `results/nodes.jsonl` 的 node_index 一致。

已核对：输入 SHA-256 正确；提取的特征、节点元数据、列名、近邻及连通性统计与既有结果完全一致；63 张图的连接结构一致，边权和 PCA 坐标在浮点容差内一致。不同环境可能改变主成分符号和舍入值。

数据包含 2000 首歌曲、673 位艺术家。32 维 PCA 的累计解释方差约为 94.42%；k=2 时分量大小为 1997 和 3，在已计算的 k=3…64 范围内全图连通。

`results/experiment.json` 保存实验参数、数据审计、运行环境和验证记录，其中 `current_pipeline` 记录处理程序及验证结果。数据来源见 `source_data/SOURCE.json`；引用和许可说明见 `source_data/CITATION.bib`、`source_data/UPSTREAM_README.md` 和 `source_data/LICENSE`。
