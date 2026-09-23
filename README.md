# ATD + Spectrum Tangent

论文 [**One Patch, Three Roles: What Is Actually Coupled in Autoregressive Time-Series Forecasting?**](https://arxiv.org/abs/2609.23686)
（arXiv:2609.23686）的代码实现。

我们研究自回归时序预测中 patch 的表示、预测转移和执行角色。
ATD 用冻结父模型的递归轨迹训练并行出口；Spectrum Tangent 用训练集选择的周期历史方向修正预测。
Atomic encoding 通过共享的小尺度原子编码与按时间顺序拼接，考察编码粒度和表示宽度的影响。

## 实现内容

| 内容 | 本仓库提供 |
| --- | --- |
| 七数据集 Patch-AR parent、ATD、Joint Direct、Frozen Direct | 模型、训练与评估入口 |
| 七数据集本地 Spectrum Tangent | 训练集选参、冻结选择、完整测试集 MSE/MAE |
| Atomic encoding | 共享原子编码与按时间顺序拼接 |
| Timer / TimesFM | 可选适配器与四数据集 campaign |
| AutoTimes | 适配器与 ETTh1 实验入口 |

## 安装

建议 Python 3.11 或更新版本，使用独立环境。根据机器选择合适的 PyTorch CPU/CUDA wheel。

```bash
git clone https://github.com/RowanFFF/ATD-Spectrum-Tangent.git
cd ATD-Spectrum-Tangent

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

AutoTimes、Timer 和 TimesFM 的依赖及模型准备见
[FOUNDATION_MODELS.md](FOUNDATION_MODELS.md)。

## 数据准备

从 [TSLib 官方说明](https://github.com/thuml/Time-Series-Library#usage)
获取标准预测数据集，解压为以下目录：

```text
dataset/
  ETT-small/ETTh1.csv
  ETT-small/ETTh2.csv
  ETT-small/ETTm1.csv
  ETT-small/ETTm2.csv
  weather/weather.csv
  electricity/electricity.csv
  traffic/traffic.csv
```

使用标准训练/验证/测试划分及仅在训练集拟合的标准化。

## 快速开始

以 ETTh1、seed 2021、ATD-4 为例，在仓库根目录依次运行：

```bash
# 1. 训练 parent，使用验证集选择 checkpoint
python scripts/paper_clean_a16_campaign.py parent --datasets ETTh1 --seeds 2021

# 2. 冻结 parent，训练 ATD-4
python scripts/paper_clean_a16_campaign.py generated --datasets ETTh1 --seeds 2021 --widths 4

# 3. 从训练 origins 选择 Tangent 周期与非负系数
python scripts/local_spectrum_tangent.py select \
  --datasets ETTh1 --seeds 2021 --widths 4 \
  --output-dir analysis_outputs/quickstart_etth1

# 4. 评估 ATD 和 ATD + Tangent
python scripts/local_spectrum_tangent.py test --open-test \
  --output-dir analysis_outputs/quickstart_etth1
```

默认使用 CUDA；Tangent 可加 `--device cpu`。
训练命令支持 `--dry-run` 预览。

输出目录包含：

- `train_moments.csv`：四个训练时间块的拟合统计；
- `selection_lock.json`：周期、gamma、有效 alpha、数据/checkpoint/代码哈希；
- `test_metrics.csv`：每个 seed、宽度和 horizon 的 ATD / ATD+Tangent MSE、MAE；
- `test_summary.csv`：跨 seed 均值与样本标准差。

测试阶段自动读取 `selection_lock.json` 中的数据集、seed 和宽度。
重复实验时请指定新的 `--output-dir`，已有结果不会被覆盖。

## 七数据集实验

默认网格：七数据集，seed 2021/2022/2023，分别训练 ATD-4 与 ATD-8，
W=672，预测 H=720，并报告 H=96/192/336/720。
配置见 [configs/paper_grid.json](configs/paper_grid.json)。

```bash
python scripts/paper_clean_a16_campaign.py parent --workers 1
python scripts/paper_clean_a16_campaign.py decoders --workers 1

# Parent、ATD 和直接监督对照的独立测试阶段
python scripts/paper_clean_a16_campaign.py test --workers 1
python scripts/paper_clean_a16_campaign.py summarize --require-complete

# 本地 ATD + Tangent：三 seed 共享每个数据集/宽度的选择
python scripts/local_spectrum_tangent.py select
python scripts/local_spectrum_tangent.py test --open-test
```

`decoders` 包含 ATD、Joint Direct、匹配的 Frozen Direct，以及额外的 frozen-placeholder 对照。
仅运行 ATD 时可改用 `generated` 阶段，再直接运行 Tangent 的配对测试；
`summarize --require-complete` 用于汇总包含全部对照的实验网格。
实验汇总位于 `analysis_outputs/exploratory/paper_clean_a16_v1_20260814/`；
Tangent 默认输出位于 `analysis_outputs/local_spectrum_tangent/`。

### 论文名称与代码名称

| 论文名称 | 入口 / 实现 |
| --- | --- |
| Parent (Recursive) | `parent` / `TransformerAblationAR` |
| ATD | `generated` / `MatchedFrozenMultiExitAR`，truth weight = 0 |
| Frozen Direct（匹配监督目标对照） | `frozen_direct_exit` / 同一模型，truth weight = 1 |
| Joint Direct | `direct_mtp_joint` / `DirectMTPAR` |
| Frozen-placeholder 对照 | `direct_mtp_frozen` / `FrozenDirectMTPAR` |
| 本地 Spectrum Tangent | `local_spectrum_tangent.py` / `PatchARSpectrumTangent.py` |

`generated` 是 ATD 在实验入口中的名称。
ATD-1 提供父模型精确回退；Tangent 对每次调用的第一个 patch 不作修改。

## 测试

运行单元测试、命令预览和发布文件检查：

```bash
make audit
```

## 致谢与许可证

本实现基于 [THUML Time-Series-Library](https://github.com/thuml/Time-Series-Library)，
上游基准提交为 `4e938a1767106324dd753b2a44832bf870a0252e`。
保留 MIT [LICENSE](LICENSE) 和 [UPSTREAM.md](UPSTREAM.md)；外部模型遵守各自许可证。
