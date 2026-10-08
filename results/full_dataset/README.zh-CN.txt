完整 MTG-Jamendo E1–E3 数据交接（2026-10-08）

本分支只新增完整数据、已验证结果与来源说明；main、原 2,000 首结果及已有 E4–E6 分析代码保持原样。
后续 E4–E6 的路径、字段和规模适配由队友负责。本次未运行或生成新 E4–E6 结果。

两个独立实验：
  moodtheme：18,486 个节点，k=2 有 67 个分量，最大分量 18,246 个节点，首个连通 k=6。
  full：55,607 个有效节点，k=2 有 113 个分量，最大分量 55,199 个节点，首个连通 k=5。
  全量标签表原有 55,609 首，track_1057640 与 track_1118596 缺少官方声学特征，已记录排除原因。
  每个实验独立拟合标准化、分组权重、32 维 PCA 和 Gaussian 带宽。标签不参与建图。

目录与数据接口：
  results/full_dataset/moodtheme/ 和 results/full_dataset/full/ 分别保存两个实验。
  graphs/k002.npz ... k064.npz：每组 63 张 float64 加权无向 union-kNN CSR 图。
  components/kXXX.npz：字段 labels、sizes；labels=0 为最大分量。
  nodes.jsonl 的 node_index 与特征、近邻、矩阵、图及分量标签逐行对齐。
  features_raw.npy：100 维 MFCC/HPCP/节奏原始特征；features.npy：32 维 PCA 特征。
  analysis_data.npz：features、neighbors、neighbor_weights；neighbor_distances.npy：近邻距离。
  preprocessing.npz：标准化、组权重、PCA 参数；feature_names.json：原始列名。
  distances.npy、similarities.npy：完整 float64 矩阵，使用 mmap_mode='r' 读取。
  source_data/full_dataset/ 保存共同原始输入与审计；metadata/ 保存官方元数据和许可；archives/ 保存全部 100 个官方分包。

数据获取：
  git clone 只获取小文件、报告、统计、校验清单和代码快照，不自动下载大数据。
  Release：https://github.com/YiJin0716/MusicGraphAnalysis/releases/tag/full-dataset-e1-e3-2026-10-08
  manifest.json 的 groups 列出分卷名称、顺序、大小及 SHA-256；files 列出恢复路径与最终文件校验值。
  先下载 handoff 分组，即可取得 E4–E6 所需的全部图、节点、特征、近邻与分量标签。
  将同一分组的 part001、part002……按顺序作二进制拼接，先核验分卷 SHA-256。
  format=raw 的分组恢复到 destination 指定的 .npy 路径。
  format=tar 的分组拼接为临时 TAR，再安全解包到仓库根目录，文件路径已按仓库目录组织。
  最后按 manifest.json 的 files 检查恢复文件 SHA-256。
  其余 dense matrix、prepared-source、official-archives 分组提供完整 E1–E3 与全部官方原始声学统计。
  不包含音频。

队友需要处理的兼容性差异：
  旧加载器读取 component_ids，新分量文件提供 labels；尚未修改加载器。
  现有脚本含固定 2,000 个节点、旧结果路径、旧分量规模及 k=3 连通假设。
  E6 的谱/节点域对照需要同一实验、同一个 k 的 E5 结果。
  使用最大分量时须保留 global_node_index 映射，不能直接将局部标签与全图标签对齐。

来源和验证：
  官方分包及 55,699 个 JSON 已按官方 SHA-256 核验。所有 126 张图与两个完整实验均已独立验证。
  config.json、validation.json 与原 artifact_manifest.json 记录原运行参数及 hash。
  code/full_dataset/snapshot/ 是未经适配的原始生成/验证代码快照，用于追溯原运行。
  原运行目录说明保存在 provenance/ORIGINAL_RUNTIME_README.txt；参考验证的 pilot 输入已存在于原仓库 source_data/tracks.jsonl.gz。
  报告导航链接已按新目录调整，原 HTML 保存在 provenance/，分发文件以 manifest.json 的 hash 为准。
  完整许可证、引用和官方元数据保存在 source_data/full_dataset/metadata/。
