MTG-Jamendo 完整数据集：两个独立 E1–E3 实验
创建日期：2026-10-08

打开 comparison.html 查看两个实验的实际结果与报告链接。
完成状态以 experiments/<name>/COMPLETE.json 与 validation.json 为准。

实验范围
  moodtheme：完整 autotagging_moodtheme.tsv 标签表，预期 18,486 首。
  full：完整 raw_30s_cleantags_50artists.tsv 清洗后标签表，预期 55,609 首。
  不采用官方 train/test split，不抽样。所有有效曲目各自参与实验。
  具体有效节点数与排除原因以 data/audit.json、data/exclusions.json 为准。
  官方 raw_30s 声学包对应更宽的 55,701 首，先完整下载并逐曲核验，再按标签表选节点。
  两个实验共用原始特征，但独立拟合标准化、分组权重、PCA 和 Gaussian 带宽。
  标签用于保留节点元数据及描述数据范围，不进入声学特征、距离或建图。

来源
  数据：https://github.com/MTG/mtg-jamendo-dataset
  已验证的参考 pipeline：https://github.com/YiJin0716/MusicGraphAnalysis
  固定 commit、下载 URL、官方校验值见 source_metadata/download_manifest.json。
  archives/ 保存全部 100 个官方 raw_30s AcousticBrainz 原始 TAR.GZ 分包。
  每个分包、每个 JSON 均验证官方 SHA-256。
  mood/theme 曲目的特征同时验证 mood/theme 官方逐曲清单，以确认共享全量包不会改变特征。
  source_metadata/ 保存官方元数据、引用、许可及校验清单。
  data/tracks.jsonl.gz 无损保留有效全量标签集所需的原始统计字段与来源。
  未下载音频；使用官方 Essentia/AcousticBrainz 预计算声学统计。

方法
  E1：100 维原始特征：13 MFCC mean/std（26），36 HPCP mean/std（72），BPM/onset_rate（2）。
      MFCC std 从协方差对角线开平方，HPCP std 从方差开平方。
      逐列 z-score（ddof=0），剔除常量列，按各组有效维数平方根缩放。
      full-SVD PCA 32 维，不白化。全部数值计算 float64。
  E2：保存完整 N×N 欧氏距离 R 与 Gaussian 相似度 W。
      sigma 为每个节点第 10 个非自身近邻距离的中位数。
      W_ij = exp(-R_ij^2/(2*sigma^2))，W_ii=0。
      分块精确计算，完整矩阵通过内存映射写盘；不使用近似近邻。
      对称块镜像写入，严格保持矩阵对称。不同 ID 的重复特征保留为独立节点。
  E3：k=2..64；距离并列按 node_index 排序；排除自身；union-kNN 对称化。
      两个实验各保存 63 张加权无向 CSR 图、逐节点分量标签与规模。
      分量编号按规模降序、最小 node_index 升序重排，labels=0 是最大分量。
      所有 k 共用同一邻居前缀与 sigma，图边集合嵌套。
      k_c 表示扫描区间 2..64 内首个连通 k；未测试 k=1。

目录
  code/                         下载、构建、验证脚本和参考代码
  source_metadata/              来源版本、元数据、官方校验值、许可
  archives/                     完整官方原始声学特征包
  data/                         核验后的特征、节点、字段 JSONL 与审计
  experiments/moodtheme/        完整 mood/theme 独立结果
  experiments/full/             全量标签集独立结果
  comparison.html/.json/.csv     对比报告与统计
  logs/                         运行日志与原 2,000 首基准验证

每个实验的主要产物
  config.json                   参数、环境版本、来源 commit 与实现 hash
  features_raw.npy               N×100 原始特征
  features.npy                   N×32 PCA 特征
  preprocessing.npz              标准化、组权重和 PCA 参数
  feature_names.json             100 个特征列名
  nodes.jsonl                    逐节点 track/artist/album ID、标签、来源
  distances.npy                  完整 N×N float64 距离矩阵
  similarities.npy               完整 N×N float64 相似度矩阵
  analysis_data.npz               features、neighbors、neighbor_weights
  neighbor_distances.npy         N×64 精确近邻距离
  graphs/kXXX.npz                SciPy CSR 图（k=2..64）
  components/kXXX.npz            labels 和 sizes，labels 与所有矩阵同一节点顺序
  connectivity.json/.csv         63 个 k 的连通性统计
  E1.json/E2.json/E3.json         各阶段实际统计
  E1_pca.png/E2_similarity.png/E3_connectivity.png
  report.html                    中文单实验报告
  artifact_manifest.json         交付文件大小与 SHA-256
  validation.json                独立验证结果
  COMPLETE.json                  阶段完成及独立验证状态

重现（从此目录执行 PowerShell；已有 Anaconda 环境满足依赖）
  python -m pip install -r code/requirements.txt
  python -u code/download.py
  python -u code/check_reference.py
  python -u code/run_experiments.py
  python -u code/validate_streamed.py
  python -u code/make_comparison_report.py
  python -u code/check_delivery.py
  构建/验证单一实验时可添加 --experiment moodtheme 或 --experiment full。
  已完整下载的分包经核验后复用；已经写入阶段完成记录的阶段不会重复计算。
  配置、代码或输入审计变动时拒绝混用既有实验目录，应复制工程到新目录再运行。
  阶段中断产生 .partial.npy，重试时重新计算该矩阵；已完成阶段不受影响。

加载（建议始终内存映射完整矩阵）
  import numpy as np
  from scipy.sparse import load_npz
  from pathlib import Path
  out = Path('experiments/moodtheme')  # 或 experiments/full
  features = np.load(out / 'features.npy')
  R = np.load(out / 'distances.npy', mmap_mode='r')
  W = np.load(out / 'similarities.npy', mmap_mode='r')
  A = load_npz(out / 'graphs/k032.npz')
  components = np.load(out / 'components/k032.npz')
  largest_indices = np.flatnonzero(components['labels'] == 0)
  A_largest = A[largest_indices][:, largest_indices]

验证
  check_reference.py 将新的分块实现与参考代码在原 2,000 首样本上逐项对照。
  近邻索引、所有 63 张图、权重、矩阵一致；另外覆盖重复点/边界并列排序。
  validate_streamed.py 顺序重读全部矩阵条目，要求每项与 float64 欧氏距离及 Gaussian 公式精确相等，并验证精确近邻排序。
  矩阵每项精确等于确定性的对称函数，因此同时证明矩阵对称性；无需随机跨列读取完整矩阵。每批后释放映射，控制驻留内存。
  validate.py 保留通过 mood/theme 全部验证的行/列交叉实现；validate_blocked.py 保留成对矩阵块实现。
  全量的前两次验证因跨行 I/O 开销主动中断，日志保存在 logs/validation_full.log 和 validation_full_blocked.log；最终结果来自 validation_full_streamed.log。
  验证 PCA 重构、节点对齐、全部 63 张图的边与权重、所有分量标签及连通性统计。
  所有交付文件生成 SHA-256；完整矩阵的校验不会一次加载到内存。

解释边界
  两组 PCA 空间与带宽不同，边权不能作为统一标尺直接跨实验比较。
  全量实验包含 mood/theme 曲目，但重新建图后的子图不等同于 mood/theme 独立图。
  缺少标签不代表属性不存在；同艺术家、重复特征可导致样本相关性。
  这里只完成用户要求的 E1–E3，后续 E4–E7 可直接读取保存的图与节点。
