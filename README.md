# 单目深度估计避障提示系统

本项目基于 KITTI Raw Dataset，完成了单目深度估计相关的数据清洗、数据集划分、统计分析、模型训练评估，以及基于实拍视频的前方障碍物提示 demo。

项目主要包含三部分：

```text
1. KITTI 数据清洗与 train / val / test 划分
2. Monodepth2 / Lite-Mono 单目深度估计训练与评估
3. 基于预测视差图的 SAFE / CAUTION / DANGER 避障提示 demo
```

当前仓库重点保留课程提交和汇报所需的核心代码、结果图表和说明文档。KITTI 原始数据、模型权重、实拍视频、TensorBoard 原始 event 日志等大文件作为本地实验材料，不纳入普通 Git 提交。

## 项目结构

```text
.
├── config.py                  # 项目路径配置
├── requirements.txt           # Python 依赖
├── scripts/                   # 数据清洗、统计作图、避障提示脚本
├── data/kitti/                # 本地 KITTI 数据与清洗结果
├── results/                   # 实验指标、loss 曲线、数据统计图
├── monodepth2/                # Monodepth2 模型代码与本地日志目录
├── lite-mono/                 # Lite-Mono 模型代码与本地实验目录
└── docs/                      # 报告、说明文档、答辩资料
```

## 环境配置

建议使用 Python 3.9+。

```bash
pip install -r requirements.txt
```

主要依赖包括：

```text
PyTorch
torchvision
NumPy
Pandas
OpenCV
Matplotlib
Pillow
scikit-image
tensorboardX
requests
```

## 数据集说明

本项目使用 KITTI Raw Dataset。

```text
使用日期：
2011_09_26
2011_09_28
2011_09_29
2011_09_30

本地路径：
data/kitti/

有效同步驾驶序列数量：
36
```

KITTI 中包含车载 RGB 图像和 LiDAR 点云。本项目先将图像与对应 LiDAR 点云按 frame_id 匹配，再进行样本有效性检查和异常样本剔除。

## 数据清洗流程

按顺序运行：

```bash
python scripts/check_kitti.py
python scripts/clean_kitti_subset.py
python scripts/analyze_kitti_cleaning.py
python scripts/split_dataset.py
python scripts/make_file_list.py
python scripts/plot_kitti_analysis.py
```

清洗规则主要包括：

```text
1. 按 frame_id 匹配 RGB 图像和 LiDAR 点云。
2. 检查图像是否可读取。
3. 检查点云是否可读取，且格式为 N x 4。
4. 记录图像尺寸和 LiDAR 点云数量。
5. 剔除 LiDAR 点数低于 100000 的异常样本。
6. 删除重复或无效样本。
7. 按驾驶序列划分训练集、验证集和测试集，避免连续帧泄漏。
```

## 数据清洗结果

```text
初始图像/点云配对样本：18201
异常样本：11
清洗后样本：18190
训练集：14109
验证集：2369
测试集：1712
清洗后 LiDAR 点数均值：122791.6
清洗后 LiDAR 点数范围：100236 - 129392
```

图像尺寸分布：

```text
370 x 1226：13240
370 x 1224：1926
374 x 1238：1556
375 x 1242：1468
```

生成的主要文件：

```text
data/kitti/kitti_subset_cleaned.csv
data/kitti/kitti_subset_abnormal.csv
data/kitti/kitti_subset_final_cleaned.csv
data/kitti/train_files.txt
data/kitti/val_files.txt
data/kitti/test_files.txt
```

生成的数据统计图：

```text
results/plots/sequence_counts_bar.png
results/plots/lidar_points_hist.png
results/plots/lidar_points_line.png
results/plots/split_pie.png
```

## 数据集划分方式

本项目采用按驾驶序列划分，而不是随机按单帧划分。

原因是 KITTI 是连续驾驶数据，相邻帧之间非常相似。如果随机按单帧划分，可能出现训练集和测试集包含几乎相同场景的情况，导致评估结果偏乐观。

划分逻辑：

```text
1. 按 sequence 分组。
2. 统计每个 sequence 的样本数量。
3. 按样本数量从大到小排序。
4. 按 train / train / val / train / train / test 的顺序轮流分配。
5. 同一 sequence 的所有帧只进入同一个集合。
```

## 模型训练与评估

本项目使用 PyTorch 训练和评估两类单目深度估计模型：

```text
Monodepth2
Lite-Mono
```

### Monodepth2

Monodepth2 使用 ResNet18 作为 encoder，并采用 ImageNet 预训练权重初始化。

需要区分：

```text
ImageNet 预训练 ResNet18：
训练开始前下载，用于初始化图像特征提取网络。

本项目训练 checkpoint：
训练后保存的 encoder.pth、depth.pth 等深度估计模型权重。
```

Monodepth2 进行了两版训练：

```text
v1：
早期较小数据规模，约几千张图像。

v2：
扩展后的数据规模，一万多张图像。
```

实验现象：

```text
v2 的训练 loss 更低，但最终验证指标没有明显优于 v1。
```

最后记录指标大致为：

```text
Monodepth2 v1:
abs_rel = 0.151586
rmse    = 5.516129
a1      = 0.812393

Monodepth2 v2:
abs_rel = 0.155694
rmse    = 6.138371
a1      = 0.790183
```

可能原因包括：KITTI 连续帧相似度较高，新增样本的数据多样性提升有限；ResNet18 模型容量有限；训练 loss 下降不一定代表深度评估指标同步提升。

### Lite-Mono

Lite-Mono 是轻量化单目深度估计模型，更适合后续边缘设备部署。

Lite-Mono 的 v1、v2、v3 主要来自训练中断后的分段和续训记录，不是三个完全不同的模型结构。通过 checkpoint 评估，综合表现较好的结果为：

```text
lite-mono-v2 / weights_6
abs_rel = 0.151
sq_rel  = 1.334
rmse    = 5.722
a1      = 0.815
```

## Loss 曲线与评估结果

主要结果文件：

```text
results/monodepth2_v1/loss_curve.csv
results/monodepth2_v1/loss_curve.png
results/monodepth2_v2/loss_curve.csv
results/monodepth2_v2/loss_curve.png
results/lite-mono/loss_curve.csv
results/lite-mono/loss_curve.png
results/lite-mono-checkpoint-eval.csv
```


```text
左图：训练损失与验证损失
右图：多尺度训练损失
```

多尺度训练损失对应 Monodepth2 在不同分辨率尺度上的输出损失，用于观察模型在多个尺度上的收敛情况。

## 单张图片推理

Monodepth2 单张图像推理示例：

```powershell
python monodepth2\test_simple.py --image_path demo\test.png --model_path monodepth2\logs\kitti_subset_v2\models\weights_39 --no_cuda
```

生成文件：

```text
demo/test_disp.jpeg
demo/test_disp.npy
```

其中：

```text
*_disp.jpeg：
视差图可视化结果。

*_disp.npy：
数值视差结果，后续避障提示逻辑会读取该文件。
```

## 避障提示 Demo

当前 demo 基于视差图进行前方风险判断，不直接识别“车”“树”“人”或“自行车”的类别。

基本逻辑：

```text
1. 输入 RGB 图像或视频帧。
2. 使用 Monodepth2 / Lite-Mono 预测视差图。
3. 选取画面中央偏下区域作为前方 ROI。
4. 统计 ROI 中近距离像素比例。
5. 输出 SAFE / CAUTION / DANGER 三种状态。
```

单张图片避障提示：

```powershell
python scripts\obstacle_warning_image.py --image_path demo\test.png
```

视频帧合成避障提示视频：

```powershell
python scripts\obstacle_warning_video.py --frames_dir demo\2_frames_30fps --output_path demo\2_warning.mp4 --ext jpg
```

常用参数：

```text
--roi_x1 / --roi_x2：
控制检测框左右范围。

--roi_y1 / --roi_y2：
控制检测框上下范围。

--near_percentile：
控制哪些像素被认为是近处。数值越低，系统越敏感。

--caution_ratio：
触发 CAUTION 的近处像素比例。

--danger_ratio：
触发 DANGER 的近处像素比例。

--layout：
控制展示样式，支持 overlay 和 triple。
```

示例：

```powershell
python scripts\obstacle_warning_video.py --frames_dir demo\2_frames_30fps --output_path demo\2_warning_near85.mp4 --ext jpg --near_percentile 85
```

说明：

```text
near_percentile = 85
表示把视差值排名前 15% 的像素视为近处，比默认值 88 更敏感。
```

## Demo 展示建议

汇报时建议展示两类场景：

```text
1. 侧边障碍物：
障碍物位于画面侧边，不在前方 ROI 内，系统提示 SAFE。

2. 正前方接近障碍物：
树木、车辆、自行车或电动车进入前方 ROI 后，系统提示 CAUTION 或 DANGER。
```

这样可以说明当前系统关注的是“前方通行区域风险”，而不是简单地看到任意物体就报警。

## 应用前景

本项目适合作为低成本视觉感知和避障提示原型，可扩展到：

```text
1. 智能辅助驾驶中的前方风险提示。
2. 移动机器人、巡检机器人、配送机器人的基础避障。
3. 视觉辅助设备中的前方障碍提醒。
4. 与红外、超声波或 LiDAR 融合，提高复杂环境下的可靠性。
```

需要注意，本项目并不是要完全替代 LiDAR、红外或超声波等传感器，而是探索一种基于普通摄像头的低成本视觉感知方案。

## 当前不足

```text
1. 单目深度估计主要反映相对深度，绝对距离不够精确。
2. 实拍视频和 KITTI 车载数据存在视角、焦距和场景分布差异。
3. 对自行车、电动车等细小稀疏障碍物的判断仍不稳定。
4. 当前避障逻辑基于固定 ROI 和阈值，规则较简单。
5. 视频 demo 目前主要采用离线逐帧处理，尚未实现实时部署。
```

## 后续改进方向

```text
1. 加入目标检测，区分车辆、行人、自行车、电动车等类别。
2. 引入视频时序平滑，减少逐帧提示抖动。
3. 根据场景自适应调整 ROI 和风险阈值。
4. 优化模型推理速度，尝试部署到 Jetson、手机或其他边缘设备。
5. 与红外、超声波或 LiDAR 进行多传感器融合。
```