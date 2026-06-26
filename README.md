# 基于 KITTI 数据集的单目深度估计模型对比研究

本项目基于 KITTI Raw Dataset，完成单目深度估计相关的数据清洗、数据集划分、统计分析、模型训练、定量评估与推理结果可视化，并对 Monodepth2 和 Lite-Mono 两类模型进行对比研究。

项目主要包含两部分：

```text
1. KITTI 数据清洗与 train / val / test 划分
2. Monodepth2 / Lite-Mono 单目深度估计训练与评估
```

当前仓库重点保留课程提交和汇报所需的核心代码、结果图表和说明文档。KITTI 原始数据、模型权重、实拍视频、TensorBoard 原始 event 日志等大文件作为本地实验材料，不纳入普通 Git 提交。

## 项目结构

```text
.
├── config.py                  # 项目路径配置
├── requirements.txt           # Python 依赖
├── scripts/                   # 数据清洗与统计作图脚本
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

## 单张图像推理

Monodepth2 单张图像推理示例：

```powershell
python monodepth2\test_simple.py --image_path assets\sample.png --model_path monodepth2\logs\kitti_subset_v2\models\weights_39 --no_cuda
```

生成文件：

```text
assets/sample_disp.jpeg
assets/sample_disp.npy
```

其中：

```text
*_disp.jpeg：
视差图可视化结果。

*_disp.npy：
数值视差结果，可用于定量分析或其他下游视觉任务。
```

## 模型对比结论

从现有测试集结果看，Monodepth2 v1 与 Lite-Mono 的综合精度接近。Monodepth2 v1 的 AbsRel 和 RMSE 略低，Lite-Mono 在部分阈值精度指标上具有竞争力，同时模型结构更轻量。Monodepth2 v2 虽然训练损失更低，但测试指标未超过 v1，说明扩大训练规模并不必然带来更好的泛化能力。

该结果表明，单目深度估计模型应同时从误差指标、阈值精度、训练成本和推理效率等维度评价，不能只依据训练 loss 选择模型。

## 应用前景

单目深度估计可作为低成本视觉感知的基础模块，可扩展到：

```text
1. 自动驾驶与机器人场景的三维环境感知。
2. 图像重建、虚拟现实和增强现实中的空间理解。
3. 轻量化模型在手机、Jetson 等边缘设备上的部署研究。
4. 与双目视觉或 LiDAR 深度信息融合，提高估计精度。
```

本项目重点研究普通摄像头条件下的深度估计方法与模型差异，不以替代高精度深度传感器为目标。

## 当前不足

```text
1. 单目深度估计主要反映相对深度，绝对距离不够精确。
2. 当前实验主要基于 KITTI 车载数据，跨数据集泛化能力尚未充分验证。
3. 不同模型的参数量、FLOPs 和端侧推理速度仍需在同一硬件环境下统一测量。
4. 训练数据包含大量连续帧，场景多样性仍有限。
5. 目前以离线推理和评估为主，尚未完成边缘设备部署测试。
```

## 后续改进方向

```text
1. 在统一测试协议下补充参数量、FLOPs、显存占用和推理速度对比。
2. 增加不同输入分辨率、平滑系数和训练轮数的消融实验。
3. 使用跨数据集测试评估模型在不同场景下的泛化能力。
4. 优化模型推理速度，尝试部署到 Jetson、手机或其他边缘设备。
5. 研究深度图边缘、纹理缺失区域和动态物体区域的误差分布。
```
