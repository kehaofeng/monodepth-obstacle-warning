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
├── scripts/                       # 数据清洗脚本
│   ├── clean_kitti_subset.py      # Step 1: 提取元数据
│   ├── analyze_kitti_cleaning.py  # Step 2: 质量分析 + 异常检测
│   ├── split_dataset.py           # Step 3: 划分 train/val/test
│   ├── export_depth_maps.py       # Step 4: LiDAR 投影生成深度图
│   ├── plot_kitti_analysis.py     # Step 5: 生成分析图表
│   ├── check_kitti.py             # 工具：快速检查数据完整性
│   ├── make_file_list.py          # 工具：从 CSV 生成文件列表
│   └── show_kitti_samples.py      # 工具：可视化样本 + 深度图
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
```

## 数据来源

- **KITTI Raw Dataset**：http://www.cvlibs.net/datasets/kitti/raw_data.php
- 当前使用 `2011_09_26` + `2011_09_28` 两个日期共 26 个同步序列
- 原始数据路径：`E:\monodepth_project\data\kitti\`

## 清洗结果

| 指标 | 数值 |
|------|------|
| 原始配对样本 | 3399 |
| 图像读取失败 | 0 |
| 点云读取失败 | 0 |
| 尺寸分布 | 375×1242 (1473), 370×1224 (1926) |
| 点云点数过低 (<100,000) | 5 |
| **清洗后样本** | **3394** |
| 训练集 | 2405 (70.9%) |
| 验证集 | 613 (18.1%) |
| 测试集 | 376 (11.1%) |

## 当前进度

- [x] 数据清洗与预处理
- [x] 深度图生成（LiDAR 投影）
- [x] 数据集划分
- [x] 统计分析与可视化
- [ ] Monodepth2 基线训练
- [ ] Lite-Mono 模型升级
- [ ] 避障提示可视化

## 团队分工

- 冯可豪 — 算法负责人，模型训练与对比实验
- 罗金天 — 数据与工程负责人，数据预处理与系统集成
- 杨梦泽 — 模型实现与调优，避障提示算法
- 于君 — 可视化与汇报，PPT 制作与演示
