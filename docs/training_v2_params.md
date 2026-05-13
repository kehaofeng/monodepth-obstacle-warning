# Monodepth2 Training V2 参数记录

## 运行命令

```bash
bash scripts/start_training.sh
```

或手动：

```bash
cd E:\monodepth2
python train.py \
  --data_path "E:\monodepth-obstacle-warning\data\kitti" \
  --split kitti_subset \
  --model_name kitti_subset_v2 \
  --png \
  --batch_size 8 \
  --height 192 \
  --width 640 \
  --num_epochs 40 \
  --num_layers 18 \
  --num_workers 2 \
  --scheduler_step_size 15 \
  --disparity_smoothness 0.0015 \
  --log_dir "E:\monodepth-obstacle-warning\logs" \
  --save_frequency 5
```

## 全部参数一览

### 数据 & 路径
| 参数 | 值 | 说明 |
|------|-----|------|
| `--data_path` | E:\monodepth-obstacle-warning\data\kitti | KITTI 原始数据路径 |
| `--split` | kitti_subset | 自定义数据集划分 |
| `--dataset` | kitti | 默认值 |
| `--png` | true | 使用 PNG 格式（原始 KITTI） |
| `--log_dir` | E:\monodepth-obstacle-warning\logs | 日志输出目录 |
| `--model_name` | kitti_subset_v2 | 模型保存目录名 |

### 模型架构
| 参数 | 值 | 说明 |
|------|-----|------|
| `--num_layers` | 18 | ResNet-18 编码器 |
| `--weights_init` | pretrained | ImageNet 预训练权重（默认） |
| `--pose_model_type` | separate_resnet | 位姿网络用独立 ResNet（默认） |
| `--pose_model_input` | pairs | 位姿输入为帧对（默认） |
| `--scales` | [0, 1, 2, 3] | 4 个尺度计算 loss（默认） |

### 训练超参数（V2 改动）
| 参数 | V1 值 | V2 值 | 说明 |
|------|-------|-------|------|
| `--num_epochs` | 20 | **40** | 训练轮数翻倍 |
| `--batch_size` | 6 | **8** | 增大 batch 减少梯度噪声 |
| `--learning_rate` | 1e-4 | 1e-4 | 不变（默认） |
| `--scheduler_step_size` | 15 | 15 | 不变，epoch 15 时 LR×0.1 |
| `--scheduler_gamma` | 0.1 | 0.1 | 硬编码，LR 衰减因子 |
| `--disparity_smoothness` | 1e-3 | **1.5e-3** | 加强平滑约束 |
| `--frame_ids` | [0,-1,1] | [0,-1,1] | 不变，中心帧 ±1 |
| `--min_depth` | 0.1 | 0.1 | 不变（默认） |
| `--max_depth` | 100.0 | 100.0 | 不变（默认） |

### 训练控制
| 参数 | 值 | 说明 |
|------|-----|------|
| `--num_workers` | 2 | DataLoader 线程数 |
| `--save_frequency` | 5 | 每 5 个 epoch 保存一次 |
| `--log_frequency` | 250 | 前 2000 步每 250 batch 记录，之后每 2000 步 |

### 消融选项（全部默认关闭）
| 参数 | 值 | 说明 |
|------|-----|------|
| `--v1_multiscale` | false | 不用 v1 多尺度 |
| `--avg_reprojection` | false | 用 min reprojection |
| `--disable_automasking` | false | 启用 auto-masking |
| `--predictive_mask` | false | 不用预测掩膜 |
| `--no_ssim` | false | 保留 SSIM |
| `--use_stereo` | false | 单目训练 |

### 学习率衰减节奏

```
Epoch  0-14: LR = 1e-4  (15 epochs, 快速学习)
Epoch 15-29: LR = 1e-5  (15 epochs, 稳定收敛)
Epoch 30-39: LR = 1e-6  (10 epochs, 精细调优)
```

### 数据集统计
- 训练集: 14,109 条
- 验证集: 2,369 条
- 每 epoch 约 1,764 个 batch（batch_size=8, drop_last=True）

### 运行环境
- Conda: monodepth
- PyTorch: 2.5.1+cu121
- GPU: RTX 4060 Laptop 8GB
- 输入分辨率: 192×640

### V1→V2 改动总结
1. Epochs 20→40：给低 LR 阶段更多调优时间
2. Batch size 6→8：减少梯度噪声，稳定 loss 曲线
3. Smoothness 1e-3→1.5e-3：加强深度图平滑约束
4. 模型名改为 kitti_subset_v2，不覆盖 V1 日志
5. Frame IDs 保持不变（用户要求）
