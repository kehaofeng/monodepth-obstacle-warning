# 单目深度估计避障提示系统

基于单目图像的深度估计与避障提示系统，使用 KITTI 数据集、Monodepth2 和 Lite-Mono 模型。

## 项目结构

```
├── data/kitti/                    # 数据交付物（清洗结果、文件列表）
│   ├── kitti_subset_cleaned.csv   # Step 1 输出：图像-点云配对元数据
│   ├── kitti_subset_cleaned.json  # 同上 JSON 格式
│   ├── kitti_subset_abnormal.csv  # Step 2 输出：异常样本清单
│   ├── kitti_subset_final_cleaned.csv  # Step 2 输出：最终清洗数据
│   ├── train_files.txt            # Step 3 输出：训练集文件列表
│   ├── val_files.txt              # Step 3 输出：验证集文件列表
│   └── test_files.txt             # Step 3 输出：测试集文件列表
├── scripts/                       # 脚本
│   ├── clean_kitti_subset.py      # Step 1: 提取元数据
│   ├── analyze_kitti_cleaning.py  # Step 2: 质量分析 + 异常检测
│   ├── split_dataset.py           # Step 3: 划分 train/val/test
│   ├── export_depth_maps.py       # Step 4: LiDAR 投影生成深度图
│   ├── plot_kitti_analysis.py     # Step 5: 生成分析图表
│   ├── convert_to_monodepth2.py   # Step 6: 转换为 Monodepth2 数据格式
│   ├── start_training.sh          # Step 7: 启动 Monodepth2 训练
│   ├── check_kitti.py             # 工具：快速检查数据完整性
│   ├── make_file_list.py          # 工具：从 CSV 生成文件列表
│   └── show_kitti_samples.py      # 工具：可视化样本 + 深度图
├── monodepth2/                     # Monodepth2 模型源码
│   ├── train.py                   # 训练入口
│   ├── trainer.py                 # 训练循环 + Loss 计算
│   ├── layers.py                  # SSIM / 视差-深度转换 / 平滑Loss
│   ├── options.py                 # 命令行参数
│   ├── kitti_utils.py             # KITTI 工具（点云加载、深度图生成）
│   ├── utils.py                   # 通用工具
│   ├── test_simple.py             # 单图推理
│   ├── evaluate_depth.py          # 深度评估
│   ├── networks/                  # 网络结构
│   │   ├── resnet_encoder.py      # ResNet 编码器
│   │   ├── depth_decoder.py       # 深度解码器
│   │   ├── pose_decoder.py        # 位姿解码器
│   │   └── pose_cnn.py            # PoseCNN（可选）
│   ├── datasets/                  # 数据加载
│   │   ├── mono_dataset.py        # 单目数据集基类
│   │   └── kitti_dataset.py       # KITTI 数据集
│   └── splits/kitti_subset/       # 训练/验证/测试文件列表
│       ├── train_files.txt
│       ├── val_files.txt
│       └── test_files.txt
├── results/plots/                 # 分析图表
│   ├── sequence_counts_bar.png    # 各序列样本数柱状图
│   ├── lidar_points_hist.png      # 点云点数分布直方图
│   ├── lidar_points_line.png      # 点云点数趋势线
│   ├── split_pie.png              # 数据集划分饼图
│   └── sample_images_grid.png     # 样本图像展示
├── requirements.txt               # Python 依赖
└── README.md
```

## 环境配置

- Python 3.9+
- 安装依赖：`pip install -r requirements.txt`

## 数据清洗流程

按顺序运行以下脚本：

```bash
# Step 1: 扫描 KITTI 原始数据，提取图像-点云配对元数据
python scripts/clean_kitti_subset.py

# Step 2: 分析数据质量，检测异常样本，输出最终清洗结果
python scripts/analyze_kitti_cleaning.py

# Step 3: 按序列划分训练/验证/测试集，生成训练文件列表
python scripts/split_dataset.py

# Step 4: 利用标定参数将 LiDAR 投影为深度图（训练 ground truth）
python scripts/export_depth_maps.py

# Step 5: 生成数据集统计图表
python scripts/plot_kitti_analysis.py

# Step 6: 转换为 Monodepth2 训练格式
python scripts/convert_to_monodepth2.py

# Step 7: 训练 Monodepth2 模型
bash scripts/start_training.sh
```

## 数据来源

- **KITTI Raw Dataset**：http://www.cvlibs.net/datasets/kitti/raw_data.php
- 当前使用 `2011_09_26` / `2011_09_28` / `2011_09_29` / `2011_09_30` 四个日期共 36 个同步序列
- 原始数据路径：`E:\monodepth-obstacle-warning\data\kitti\`

## 清洗结果

| 指标 | 数值 |
|------|------|
| 原始配对样本 | 18201 |
| 图像读取失败 | 0 |
| 点云读取失败 | 0 |
| 尺寸分布 | 375×1242 (1473), 370×1224 (1926), 374×1238 (1556), 370×1226 (13246) |
| 点云点数过低 (<100,000) | 5 |
| 点云文件缺失 (drive_0009) | 4 |
| 其他异常 | 2 |
| **清洗后样本** | **18190** |
| 训练集 | 14109 (77.6%) |
| 验证集 | 2369 (13.0%) |
| 测试集 | 1712 (9.4%) |

## 数据分析结论

### 1. 序列分布（柱状图）
36 个序列的样本数差异较大：最大为 `drive_0028`（5171 帧）和 `drive_0018`（2762 帧），最小仅 29~35 帧。整体以短序列为主，但 2011_09_30 包含多个超长序列（>1000 帧），说明 KITTI 采集覆盖了不同时长的驾驶场景。划分时按序列整体分配，避免了同一场景下的相似帧泄露到多个集合。

### 2. LiDAR 点云分布（直方图 + 折线图）
点云点数集中在 118,000~122,000 区间，均值约 120,554，标准差 4,711，整体分布较为集中。仅 5 个样本点数低于 100,000（最低 98,322），可能由场景空旷或传感器瞬时遮挡导致，已予以剔除。点云密度适中，投影后可获得较完整的深度监督信号。

### 3. 图像尺寸（分布统计）
存在四种分辨率：`375×1242`（2011_09_26）、`370×1224`（2011_09_28）、`374×1238`（2011_09_29）、`370×1226`（2011_09_30）。这是不同采集批次使用的相机型号差异所致。训练时需统一缩放到固定分辨率（如 640×192）。

### 4. 数据质量
无图像/点云读取失败、无重复帧、无尺寸异常。数据质量良好，可直接用于模型训练。

## Monodepth2 模型训练

```bash
# 1. 转换数据为 Monodepth2 格式
python scripts/convert_to_monodepth2.py

# 2. 启动训练（默认 20 epochs, batch_size=6）
bash scripts/start_training.sh

# 自定义参数
bash scripts/start_training.sh 30 8
```

训练日志和模型权重保存在 `logs/kitti_subset_model/`，使用 TensorBoard 监控：

```bash
tensorboard --logdir logs
```

### 模型推理

```bash
cd monodepth2
python test_simple.py \
  --image_path <图片路径> \
  --model_path logs/kitti_subset_model/models/weights_19
```

输出：视差图 `.jpeg` 和 `.npy` 文件。

### 训练结果（20 epochs，扩展数据集）

| 指标 | 数值 |
|------|------|
| 训练样本 | 14109 |
| 验证样本 | 2369 |
| 测试样本 | 1712 |
| 输入分辨率 | 192×640 |
| 编码器 | ResNet18 (pretrained) |
| Batch size | 6 |
| 最终模型 | weights_19 |

验证集最佳指标：

| 深度误差 | 数值 |
|------|------|
| AbsRel | 0.101 |
| SqRel | 0.671 |
| RMSE | 4.539 |
| RMSE(log) | 0.165 |

| 深度准确率 | 数值 |
|------|------|
| δ<1.25 | 0.812 |
| δ<1.25² | 0.938 |
| δ<1.25³ | 0.971 |

## 当前进度

- [x] 数据清洗与预处理
- [x] 深度图生成（LiDAR 投影）
- [x] 数据集划分
- [x] 统计分析与可视化
- [x] Monodepth2 基线训练
- [ ] Lite-Mono 模型升级
- [ ] 避障提示可视化

## 团队分工

- 冯可豪 — 算法负责人，模型训练与对比实验
- 罗金天 — 数据与工程负责人，数据预处理与系统集成
- 杨梦泽 — 模型实现与调优，避障提示算法
- 于君 — 可视化与汇报，PPT 制作与演示
