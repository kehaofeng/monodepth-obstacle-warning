# 项目代码审查报告

## 1. 硬编码 Windows 路径（阻断性问题）

所有脚本和数据文件都硬编码了 `E:\monodepth-obstacle-warning\`，clone 后无法在其他机器上运行。

**受影响的脚本（11 个）：**

| 文件 | 行号 |
|------|------|
| `scripts/clean_kitti_subset.py` | 8-9 |
| `scripts/analyze_kitti_cleaning.py` | 7-9 |
| `scripts/split_dataset.py` | 11-12 |
| `scripts/export_depth_maps.py` | 16-18 |
| `scripts/plot_kitti_analysis.py` | 10-11 |
| `scripts/convert_to_monodepth2.py` | 12-13 |
| `scripts/make_file_list.py` | 6-7 |
| `scripts/show_kitti_samples.py` | 12-14 |
| `scripts/check_kitti.py` | 8 |
| `scripts/download_kitti.py` | 13 |
| `scripts/start_training.sh` | 14-18（含 Anaconda 路径） |

**受影响的数据文件：**
- `data/kitti/train_files.txt`
- `data/kitti/val_files.txt`
- `data/kitti/test_files.txt`
- `data/kitti/kitti_subset_cleaned.csv`
- `data/kitti/kitti_subset_abnormal.csv`
- `data/kitti/kitti_subset_final_cleaned.csv`
- `data/kitti/kitti_subset_cleaned.json`

**修复方案：** 用 `argparse` 或统一配置文件 `config.py` 管理路径，让用户通过命令行或环境变量指定数据目录。

---

## 2. 缺少 LICENSE 文件（合规风险）

`monodepth2/` 下所有 `.py` 文件头部均声明：

> the full terms of which are made available in the LICENSE file

但仓库中不存在 LICENSE 文件。Monodepth2 原始代码来自 Niantic，受非商用许可限制。

**修复方案：** 从 [Monodepth2 原仓库](https://github.com/nianticlabs/monodepth2) 复制 LICENSE 文件到项目根目录。

---

## 3. requirements.txt 严重不全

**缺少的核心依赖：**

| 缺失的包 | 引用位置 |
|----------|----------|
| `torch` | 所有模型文件 |
| `torchvision` | `test_simple.py`, `resnet_encoder.py` |
| `tensorboardX` | `trainer.py`（TensorBoard 日志写入） |
| `scikit-image` | `kitti_dataset.py`（`skimage.transform`） |
| `Pillow` | `mono_dataset.py`, `kitti_dataset.py`, `test_simple.py` |
| `requests` | `download_kitti.py` |

**未使用的依赖：**

| 多余的包 | 说明 |
|----------|------|
| `tqdm` | 全代码库无任何 `import tqdm` |

**修复方案：** 建议的完整 requirements.txt：

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
pandas>=1.3.0
opencv-python>=4.5.0
matplotlib>=3.4.0
Pillow>=9.0.0
scikit-image>=0.19.0
tensorboardX>=2.6.0
requests>=2.28.0
```

---

## 4. README 与实际数据不一致

- README 称使用了 4 个日期（2011_09_26/28/29/30），但 `download_kitti.py` 和 `.gitignore` 中包含第 5 个日期 **2011_10_03**
- `data/kitti/test_files.txt` 大小为 3.1 MB，远大于 `val_files.txt`（0.5 MB），与 README 中"验证集 2369、测试集 1712"的说明矛盾
- README 中说 KITTI 数据路径为 `E:\monodepth-obstacle-warning\data\kitti\`，这对 Mac/Linux 用户无意义

---

## 5. .gitignore 问题

### 5.1 未排除非可移植数据文件

以下文件包含绝对 Windows 路径，但已被 git 跟踪：

```
data/kitti/*_files.txt
data/kitti/*.csv
data/kitti/*.json
```

这些文件因机器而异，不应被版本控制。

### 5.2 KITTI 原始数据排除方式不灵活

当前逐个列出日期目录，建议用通配符代替：

```gitignore
data/kitti/2011_*/
data/kitti/depth_maps/
```

### 5.3 缺少常见 Python 忽略模式

```gitignore
*.egg-info/
dist/
build/
.mypy_cache/
.pytest_cache/
*.so
*.egg
*.key
*.pem
credentials*
```

---

## 6. 代码质量问题

### 6.1 未使用的导入

- `trainer.py:26` — `from IPython import embed`（调试遗留）
- `test_simple.py:21` — `from torchvision import datasets`（未使用）

### 6.2 硬编码 `.cuda()` 调用

`trainer.py:458` 和 `evaluate_depth.py:96-98` 中写了 `.cuda()`，在无 GPU 机器上会崩溃。应统一使用 `self.device`。

### 6.3 变量遮蔽

`mono_dataset.py:100-101`：
```python
n, im, i = k
for i in range(self.num_scales):  # i 被遮蔽
```
内部循环变量 `i` 覆盖了从元组解包得到的 `i`。虽然当前未引发 bug，但极易出错。

### 6.4 `nn.BCELoss()` 重复实例化

`trainer.py:458` 每次调用都 `nn.BCELoss()(...)`，应在 `__init__` 中创建一次并复用。

### 6.5 KITTI 数据加载无异常处理

`kitti_utils.py:12` 的 `np.fromfile()` 未包裹 try/except，损坏文件将导致程序崩溃。

---

## 7. 数据目录冗余

存在两份数据集划分文件：

| 位置 | 格式 |
|------|------|
| `monodepth2/splits/kitti_subset/` | Monodepth2 格式（`date/seq frame_id side`） |
| `data/kitti/` | 绝对 Windows 路径 |

训练代码使用前者，后者不应被提交。

---

## 8. 平台可移植性

- `start_training.sh` 中写了 `D:/anaconda3/envs/monodepth/python.exe`，仅 Windows 可用
- CSV 写入使用 `encoding="utf-8-sig"`（BOM），在 Linux/Mac 可能产生多余字符
- 核心 `monodepth2/` 代码本身是跨平台的，问题集中在 `scripts/` 和 `data/`

---

## 修复优先级

| 优先级 | 问题 | 影响 |
|--------|------|------|
| P0 | 硬编码路径 → 可配置 | 项目不可复现 |
| P0 | 补全 requirements.txt | 无法安装运行 |
| P1 | 添加 LICENSE | 合规风险 |
| P1 | 修正 .gitignore | 仓库含非可移植数据 |
| P2 | 修正 README 数据描述 | 误导使用者 |
| P2 | 清理代码质量问题 | 潜在 bug |
| P3 | 统一数据目录 | 代码整洁 |
