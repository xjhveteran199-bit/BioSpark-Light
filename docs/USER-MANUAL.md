# BioSpark-Light 用户手册

> **文档版本**:对应 BioSpark-Light v0.2.0
> **适用对象**:第一次拿到 BioSpark-Light 想完整跑通"从原始数据到训练好的模型"的所有人。
> **English version**: see [USER-MANUAL.en.md](USER-MANUAL.en.md).

---

## 目录

1. [这是什么](#1-这是什么)
2. [安装与启动](#2-安装与启动)
3. [整体工作流](#3-整体工作流)
4. [Data Prep 标签页](#4-data-prep-标签页)
5. [Train 标签页](#5-train-标签页)
6. [训练结果(T5)怎么看](#6-训练结果t5怎么看)
7. [My Models 标签页](#7-my-models-标签页)
8. [完整案例:5 手势 OpenBCI EMG 识别](#8-完整案例5-手势-openbci-emg-识别)
9. [核心概念速查](#9-核心概念速查)
10. [数据采集建议(从单 session 到可发表)](#10-数据采集建议)
11. [常见问题与故障排查](#11-常见问题与故障排查)
12. [文档地图](#12-文档地图)

---

## 1. 这是什么

**BioSpark-Light** 是 [BioSpark](https://github.com/...) 的本地桌面精简版——
一个**离线、零注册、零联网**的生物信号训练台。

**它能做的**:

- ✅ 把原始长录音(CSV / TXT / OpenBCI 导出)切成可训练的样本
- ✅ 训练 1D-CNN 分类器(ECG / EEG / EMG / 任何时序信号)
- ✅ 自动绘制混淆矩阵 / t-SNE / 训练曲线 / 发表级 figures
- ✅ 帮你**主动发现数据泄漏**(自动 group-aware 切分 + 黄色警告 banner)
- ✅ 暖启动:每次训练在前一个最佳 checkpoint 上继续

**它不做的**:

- ❌ 在线推理服务(用大版 BioSpark)
- ❌ 实时信号流监控(同上)
- ❌ Grad-CAM / SHAP 之类的 XAI(已有版本里阉掉了)
- ❌ 多用户协作(单机单用户设计)
- ❌ 替你思考数据采集协议(本手册 §10 给指导)

---

## 2. 安装与启动

### 2.1 普通用户:双击 .exe(推荐)

1. 从 [GitHub Releases](https://github.com/xjhveteran199-bit/BioSpark-Light/releases) 下载 `BioSpark-Light-vX.X.X.zip`
2. 解压到任意目录(**不要解压到 Desktop**,Windows 杀毒软件可能误报)
3. 双击 `BioSpark-Light.exe`
4. 控制台出现 `[biospark.startup] BioSpark-Light ready`,浏览器自动打开 <http://127.0.0.1:8765>
5. 系统托盘出现一个图标,右键 → Quit 退出

### 2.2 开发者:源码运行

```bash
git clone https://github.com/xjhveteran199-bit/BioSpark-Light.git
cd BioSpark-Light
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
python launcher.py
```

详见 [BUILD.md](BUILD.md)。

### 2.3 数据存哪

| OS | 路径 |
|----|------|
| Windows | `%LOCALAPPDATA%\BioSpark-Light\` |
| macOS | `~/Library/Application Support/BioSpark-Light/` |
| Linux | `~/.local/share/BioSpark-Light/` |

里面有:

```
biospark.db              # SQLite,训练记录元数据
uploads/                 # 你上传的原始数据集
checkpoints/0/v1.pt      # 训练好的模型(版本链)
              v2.pt
```

**重装应用、移动 .exe 都不会丢数据**——数据是按用户存的,跟 .exe 位置无关。

---

## 3. 整体工作流

```
┌───────────────────┐     ┌───────────────────┐     ┌────────────────────┐
│  原始数据         │     │                   │     │   一份训练-ready   │
│  (CSV/TXT/ZIP)   │ ──► │  📦 Data Prep     │ ──► │   CSV(含 __group__)│
│  长录音 / 多文件  │     │  Mode A / B / C   │     │                    │
└───────────────────┘     └───────────────────┘     └────────┬───────────┘
                                                              │
                                                              ▼
                          ┌───────────────────┐     ┌────────────────────┐
                          │  📊 T5 结果可视化 │     │  🎯 Train          │
                          │  混淆矩阵 + t-SNE │ ◄── │  Smart Auto / 自定义│
                          │  泄漏 banner      │     │  WebSocket 实时流  │
                          └───────────────────┘     └────────┬───────────┘
                                                              │
                                                              ▼
                                                    ┌────────────────────┐
                                                    │  🗂 My Models     │
                                                    │  版本链 + 暖启动   │
                                                    └────────────────────┘
```

**三条铁律**:

1. **始终走 Data Prep**——不要直接把切好的 CSV 喂 Train(手册 §4 解释为什么)
2. **结果页有黄色 banner 不要忽略**——这是诚实信号,§6.3 详解
3. **报告测试集准确率,不报告验证集准确率**——`Evaluation set: held-out test set` 才是真数字

---

## 4. Data Prep 标签页

把"长录音"切成"训练样本"。**所有数据都要先过这里**。

### 4.1 选哪个 Mode

| Mode | 适合的输入 | 例子 |
|------|------------|------|
| **A — Folder-per-class ZIP** | 文件夹名 = 类标签,每个文件是一段独立录音 | `gestures.zip` 里 `CLENCH/r1.csv`、`CLENCH/r2.csv`、`FIVE/r1.csv` ... |
| **B — Single CSV/TXT + intervals** | 一段连续录音,你手动指定时间区间和标签 | 一段 30 分钟的实验录音,你说"0–600s 是 CLENCH,600–1200s 是 FIVE..." |
| **C — Generic ZIP + per-file label** | 多个文件平铺,每个文件你手动给标签 | `data.zip` 里 `LIU-CLENCH.txt`、`LIU-FIVE.txt`...(本手册案例就是这种) |

> **不知道选哪个?** 看你的数据是不是已经按"类别文件夹"组织好。是 → Mode A;一段 CSV 自己分段 → Mode B;别的所有情况 → Mode C。

### 4.2 切分参数(每个 Mode 都有)

| 字段 | 说明 | 典型值 |
|------|------|--------|
| **Sampling Rate (Hz)** | 信号采样率 | 250(OpenBCI Cyton) |
| **Segment Length (sec)** | 每个训练样本对应的时长 | 1.0(EMG)、2.0(EEG epoch)、按需 |
| **Overlap Ratio** | 相邻 segment 的重叠比例,**有 stride 时此项失效** | 默认 0 |
| **Signal Column Index** | 单通道时用的列号(0-based) | OpenBCI 文件通常用多通道字段代替它 |
| **Multi-channel Columns** | 多通道列范围,接受 `1-5` 或 `1,2,3,4,5` | OpenBCI Cyton:`1-8`(8 通道全用)或 `1-5`(只用前 5) |
| **Stride (sec, optional)** | 两段之间的起点距离。留空 → 按 overlap 切 | 「每 5s 做 1s 任务」: `segment=1, stride=5` |
| **Group-aware split unit** | 切分粒度: `recording`(默认) 或 `trial` | 详见下方 |

#### Group-aware split unit 选哪个?

> **决定后续 train/val/test 切分时,什么算"一组"。组级切分能避免数据泄漏。**

| 你的场景 | 选 | 为什么 |
|---------|----|------|
| 每个手势采集了**多段独立录音**(多次贴电极、多个 session) | **`recording`(默认)** | 一个文件 = 一组。同一段录音的所有窗自动绑在同一切分,不会"半段在 train、半段在 test" |
| 每个手势只有 **1 段长录音**,里面有「任务-休息-任务-休息」**重复 trial** | **`trial`** | 把每个截出来的 1s 任务当独立组。如果选 recording,GroupShuffleSplit 会把整个类丢到一份切分,test 只剩 1 类 |
| 已经预切好的 CSV 没有 `__group__` 列 | 任意,Prep 也救不了 | 这种数据应该回到原始长录音重新过 Prep |

**简单粗暴决策树**:你**每个 class 的文件数 ≥ 4** ? → `recording`。否则 → `trial`。

### 4.3 OpenBCI 自动检测

上传 `.txt` 时,后台会扫前 4 KB 找 OpenBCI 头部。**检测到的话**:

- ✅ `Sample Rate = 250.0 Hz` → 自动填到 SR 字段
- ✅ 自动跳过 SampleIndex 列(0-255 循环整数检测)
- ✅ 自动跳过饱和列(99% 都是 -187500.02 的未接通 EXG)
- ✅ 自动跳过零方差列(Accel 三轴未动)
- ✅ 多通道列字段自动填 `1-N`(剩下的真实信号通道)

→ 你只需要填 segment / stride / group_by 三项。

### 4.4 文件 → 标签映射(Mode B / Mode C 必填)

下方有个表格,每行一个文件,**「标签」框留空 = 跳过这个文件**。Mode C 至少要给一个文件填标签,否则报错。

类标签可以**用任何字符串**:`CLENCH`、`握拳`、`gesture_01` 都行,会原样传到训练里。

### 4.5 输出 CSV 的 schema

`Generate Training CSV` 成功后,生成的 CSV 长这样(单通道情况):

```
s1, s2, s3, ..., sN, label, __group__
0.12, 0.34, ..., 0.56, CLENCH, l/LIU-CLENCH.txt#trial_0
0.13, 0.35, ..., 0.57, CLENCH, l/LIU-CLENCH.txt#trial_1
...
```

多通道情况下列名变成 `ch1_1, ch1_2, ..., ch1_S, ch2_1, ..., chC_S`,trainer 会自动检测并 reshape 成 `(N, C, L)` 输入 1D-CNN。

**`__group__` 列不进模型**——它是给 train/val/test 切分用的。

---

## 5. Train 标签页

### 5.1 上传或跳过

如果你刚从 Prep 点了 **「用于训练」**,这里数据集已经载入。
否则可以单独上传一个已经准备好的训练 CSV(带 `__group__` 最佳)。

### 5.2 训练模式 preset

| Preset | 用法 | 时长 | 备注 |
|--------|------|------|------|
| **🚀 Smart Auto** | 默认推荐,新手用这个 | 3–10 min CPU | 自动选架构 + 类别加权 + 早停 |
| **⚡ Quick Test** | 调参中检查数据有没有问题 | 30s–2 min | 20 epochs,跳过 LR 搜索 |
| **🏆 Publication Ready** | 投稿前最后一跑 | 15–45 min | 100 epochs + LR 搜索 |
| **⚙️ Custom** | 全手动 | 取决于设置 | 想完全控制时用 |

### 5.3 模型架构选择(v0.2 新)

数据上传后,质量评分卡片下方会出现 **「模型架构推荐」** 区块。系统按你的数据自动从两种架构里二选一:

| 架构 | 实现 | 参数量 | 适用 |
|------|------|--------|------|
| **1D-CNN** | `Signal1DCNN` | ~44K | 小样本 / 短序列 / 单通道 |
| **1D-CNN + Transformer 混合** | `Signal1DCNNTransformer` | ~350K | 长序列 / 多通道 / 样本充足 |

**推荐规则**(置信度同时给出):
- 样本数 < 500 或被试组数 < 5 → 推荐 CNN(混合模型小样本易过拟合)
- 序列长度 < 128 → 推荐 CNN(没什么长程依赖可建模)
- 样本数 ≥ 500 + 序列 ≥ 256 + 多通道 → 推荐混合
- 样本数 ≥ 2000 → 推荐混合
- 其他 → 默认 CNN(保守)

下方三个单选:**自动(系统推荐)** / **1D-CNN** / **1D-CNN + Transformer 混合**。如果你对自己的数据更有把握,可以手动切到任一选项;切到非系统推荐时会出现「⚠ 已覆盖系统推荐」标记,训练完成后也会记录到 job metadata,方便论文里写消融。

### 5.4 训练前的两个 banner(都是黄色警告)

| Banner | 含义 | 对策 |
|--------|------|------|
| ⚠️ 此数据集没有 `__group__` 列 | 你直接传了切好的 CSV,跳过了 Prep | 回到 Prep 重做 |
| (Overlap > 0 时显示) | 你设了 overlap_ratio > 0 | 多数情况下设 0,有把握再开 |

### 5.5 训练配置(只在 Custom 显示)

通常不用动,但如果是 Custom:

| 字段 | 默认 | 何时改 |
|------|------|--------|
| 训练轮数 | 30–50 | 数据集很小可以减少 |
| 学习率 | 0.001 | Smart Auto 会自动找 |
| 批大小 | 64 | 显存不够减半 |
| 验证集比例 | 0.2 | 小数据集可以加到 0.3 |
| **通道数** | 0(自动) | 列名前缀有 `ch{N}_M` 时自动检测 |
| 早停耐心值 | 10 | 数据集小可以减到 5–8 |
| 自动类别权重 | ✓ | 类别极度不平衡时启用 |
| 搜索最优学习率 | ✗ | Publication Ready 才开 |
| **从上一个模型继续训练**(暖启动) | ✗ | 见 §7 |

### 5.6 启动训练 → 看日志第一行

点 **开始训练**。WebSocket 一连上,Epoch 日志的第一行会显示**切分模式**:

```
✅ 绿色: 切分: 组级切分 · 305 个录音组 · train=183 val=61 test=61
⚠️ 黄色: 切分: 逐行随机切分（缺少 __group__ 列） · train=180 val=60 test=60
```

看到黄色的话,马上停掉,回 Prep 重做(或者数据本来就不能 group split)。

### 5.7 训练曲线读法

| 模式 | Train Loss | Val Loss | 解读 |
|------|-----------|----------|------|
| 健康 | 持续下降 | 跟着下降然后稳定 | ✅ 正常,跑完就行 |
| 过拟合 | → 0 | 早期下降,后面反弹 | ⚠️ 早停应该已经触发,如果没就手动停 |
| 学习不动 | 抖动不下降 | 抖动不下降 | ⚠️ 检查通道数 / 数据是否归一化奇怪 |
| 数据泄漏 | 几个 epoch 就 99%+ | val 跟着 99%+ | ⚠️ 训完看 §6.3 banner |

---

## 6. 训练结果(T5)怎么看

训练完成后页面自动滚到 **T5. 训练结果可视化**,有两块:

### 6.1 混淆矩阵

**矩阵上方一行小字**:

```
Evaluation set: 独立测试集 (held-out test set)
```

✅ 这一行**必须是 test**。如果是 `validation set (no held-out test)`,说明你的数据集太小,test 集被自动放弃了——这种情况下数字不可信。

**轴**:行 = 真实类(`True`),列 = 预测类(`Predicted`)。对角线是分对的,非对角线是分错的。

**显示模式切换**:左上角的 `#→` / `%→` 按钮在数字 / 百分比之间切换。

### 6.2 t-SNE 特征可视化

把模型倒数第二层(128 维特征)用 t-SNE 投影到 2D。**正常情况下应该看到的**:

- 5 个簇大致分得开
- **簇间有适度重叠**(尤其是混淆矩阵上互相错的那两类)
- 每个簇内部紧凑但不是"一个点"

**异常信号**:5 个簇分得**过分干净**(几乎不重叠) + 混淆矩阵全 100% → **数据泄漏**(详见 §9)。

### 6.3 ⚠️ 黄色「疑似数据泄漏」banner

**触发条件(同时满足)**:

1. 整体准确率 ≥ 99%
2. 任一类的 test 支持数 < 30
3. 数据集没有 `__group__` 列 **或** 没有独立 test 集

**触发后怎么办**?banner 里的 `reason` 字段会告诉你**具体哪一条触发的**:

| reason 内容关键词 | 你应该做的 |
|------|---------|
| `does not carry recording-level group ids` | 你跳过了 Prep。回 Prep 重做 |
| `there is no held-out test set` | test 集太小被放弃了。数据增多再训 |
| `Test accuracy is ≥99% on only N samples` | test 集小到统计噪声大,数字不可信 |

### 6.4 Per-class metrics 表格

每个类的 precision / recall / F1 / support。**最该看的是 support 列**——它告诉你每个类在 test 集里有多少样本。**< 30 就别认真对待这个类的数字**。

---

## 7. My Models 标签页

### 7.1 版本链

每次训练成功后,会在数据库里存一条 `TrainingRun` + 一个 `ModelCheckpoint`(`v1.pt`、`v2.pt`、...)。这里能看到:

- 历史训练记录(时间、最佳 val_acc、preset、暖启动来源)
- 当前激活的 checkpoint(下次训练默认从它暖启动)
- 每个版本的输入 shape(`n_channels` / `n_classes` / `signal_length`)

### 7.2 暖启动(Self-Improving)

下次训练在 Train 标签页勾上 **「从我的上一个模型继续训练」**,trainer 会:

1. 找到最新的兼容 checkpoint(channels 数匹配)
2. 加载它的特征提取器权重(`features.*`)
3. 重新初始化分类头(`classifier.*`,因为类数可能变)
4. 在你的新数据上 fine-tune

**特别注意**:暖启动**会复制泄漏问题**——如果你 v10 是泄漏训出来的(99%+),v11 暖启动后开局还是带"假记忆",fine-tune 几个 epoch 就又能"99%"。

> **看到「已找到兼容 vN(验证准确率 0.0%)」是怎么回事?** 那是上一次中断/失败的训练。**绝对不要勾暖启动**,会污染本轮。

---

## 8. 完整案例:5 手势 OpenBCI EMG 识别

> 这一节把 §1–§7 串起来,用一份**真实数据**走完整流程。所有数字都跟你直接复现。

### 8.1 协议背景

| 项 | 值 |
|---|---|
| 受试者 | LIU(单人) |
| 手势 | CLENCH(握拳) / FIVE(比五) / OK / ROCK(摇滚) / TWO(比二) |
| 设备 | OpenBCI Cyton 8 通道,实际接了前 5 个 |
| 采样率 | 250 Hz |
| 每手势时长 | 5 分钟连续录音 |
| 任务结构 | 「**5 秒一段,前 1 秒是有效任务,后 4 秒休息**」× 60 段 |
| 数据格式 | OpenBCI GUI 导出的 `.txt`(`%`-头 + 13 列数据) |

数据包结构:

```
l.zip
└── l/
    ├── LIU-CLENCH.txt   ← 9.5 MB,~75 000 行
    ├── LIU-FIVE.txt
    ├── LIU-OK.txt
    ├── LIU-ROCK.txt
    └── LIU-TWO.txt
```

5 个文件**平铺**在一个文件夹下,**不是 folder-per-class** → 必须用 **Mode C**。

### 8.2 步骤 1 — 启动应用

双击 `BioSpark-Light.exe`,浏览器打开。

### 8.3 步骤 2 — Data Prep

#### (a) 上传

1. 切到 **Data Prep** 标签
2. 选 **Mode C — generic ZIP**
3. 拖 `l.zip` 进上传框

inspect 跑完后,自动检测到 OpenBCI 头并填好:

| 字段 | 自动填 |
|------|--------|
| Sampling Rate (Hz) | 250 |
| Multi-channel Columns | **`1-5`** ← 自动跳过 SampleIndex / 饱和 / Accel |

#### (b) 手动改 3 项

| 字段 | 改成 | 为什么 |
|------|------|--------|
| Segment Length (sec) | **`1`** | 每段前 1s 是有效任务窗口 |
| Stride (sec, optional) | **`5`** | 每 5s 一段,跳过中间 4s 休息 |
| **Group-aware split unit** | **`Trial 级`** | 一类只有 1 段录音 + 60 个 trial,必须 trial 级 |

(Overlap Ratio 和 Signal Column Index 不动,默认值就是对的)

#### (c) 文件 → 标签映射

| File | Class label |
|------|-------------|
| `l/LIU-CLENCH.txt` | `CLENCH` |
| `l/LIU-FIVE.txt`   | `FIVE` |
| `l/LIU-OK.txt`     | `OK` |
| `l/LIU-ROCK.txt`   | `ROCK` |
| `l/LIU-TWO.txt`    | `TWO` |

#### (d) 点 「Generate Training CSV」

期望输出:

```
Generated 305 samples across 5 classes.
```

| 维度 | 值 |
|------|-----|
| 样本总数 | 305(每类 61 段,5min 多录了一点 → 60.4 → 61) |
| 信号长度 | 250(= 1s × 250Hz) |
| 通道数 | 5(自动检测) |
| `__group__` | 305 个独立 trial id |
| 列总数 | 1252(`ch1_1..ch5_250` + `label` + `__group__`) |

预览表里前 10 行**都是 CLENCH 是正常的**——CSV 按文件顺序拼接,前 61 行都是 CLENCH。Trainer 内部会洗牌,不影响训练。

点 **用于训练**,自动跳到 Train 标签页。

### 8.4 步骤 3 — Train

#### (a) 检查

数据预览下方**应该没有黄色 banner**。如果有 → 回 Prep 重做。

#### (b) 配置

| 项 | 选 / 填 |
|---|---|
| Preset | **🚀 Smart Auto** |
| Number of channels | 留空 或 填 `5` |
| 自动优化 | ✓ |
| 自动类别权重 | ✓ |
| 搜索最优学习率 | ✗ |
| **从上一个模型继续训练** | **✗(关键!)** |

#### (c) 点 「开始训练」

第一行 epoch 日志应该是**绿色**的:

```
✅ 切分: 组级切分 · 305 个录音组 · train=183 val=61 test=61
```

3–10 分钟内训完。

### 8.5 步骤 4 — 看结果

T5 训练结果可视化里:

#### 混淆矩阵(实际跑出来的数字)

|  | CLENCH | FIVE | OK | ROCK | TWO |
|---|---|---|---|---|---|
| **CLENCH** | **91.7%** | 0% | 8.3% | 0% | 0% |
| **FIVE** | 0% | **100%** | 0% | 0% | 0% |
| **OK** | 11.1% | 0% | **88.9%** | 0% | 0% |
| **ROCK** | 0% | 0% | 0% | **100%** | 0% |
| **TWO** | 0% | 0% | 0% | 0% | **100%** |

整体准确率约 **96.6%**(57/59),**没有触发**黄色泄漏 banner。

#### 这次为什么是健康的?

| 之前(泄漏)| 这次(诚实) |
|---|---|
| 5 类全 100%(对角线无误差) | CLENCH↔OK 互混 ~10% |
| t-SNE 5 个簇分得开 4–10 个单位 | 簇间有适度重叠 |
| 触发黄色泄漏 banner | banner 不触发(因为 < 99%) |

CLENCH↔OK 的混淆**生物学上 make sense**——握拳和 OK 手势都涉及拇指 + 食指弯曲,前臂浅层屈肌(EXG 1-5 大致采集这片)电活动相似。模型学的是真实肌电模式区分,不是时序记忆。

### 8.6 这个 96.6% 能不能发表?

**不能**。原因:

| 维度 | 现状 | 论文要求 |
|---|---|---|
| 受试者 | 1 人 | ≥ 5 人 |
| Session 数 | 1 / 类 | ≥ 4 / 受试者 |
| Test 集大小 | ~12 / 类 | ≥ 30 / 类 |
| 切分方式 | trial 级 | subject 级 LeaveOneOut |

**能说**:"BioSpark-Light pipeline 在受试者 LIU 单 session 数据上达到 96.6% test 准确率"。
**不能说**:"本方法对 5 类手势分类准确率 96.6%"(暗示泛化)。

跨 session 的可发表数字怎么得到?见 §10。

---

## 9. 核心概念速查

### 9.1 数据泄漏 (Data Leakage)

> **同一段录音切出来的窗,既出现在 train 又出现在 val/test** → 模型背答案,准确率虚高。

防御方法 = 组级切分(group-aware split):

```python
sklearn.model_selection.GroupShuffleSplit
```

**BioSpark-Light 默认开启**,前提是数据有 `__group__` 列(Prep 自动生成)。

### 9.2 `__group__` 列

每个样本的"录音/trial id",训练时用来切分,**不进模型**。

| Prep 模式 | `__group__` 写什么 | 由 group_by 控制? |
|---|---|---|
| Mode A | 文件路径 | ✓ |
| Mode B | 区间索引 `interval_0`、`interval_1`... | ✓ |
| Mode C | 文件名 | ✓ |

`group_by=trial` 时会变成 `<原值>#trial_<i>`,每段独立。

### 9.3 group_by 选什么

| 数据形态 | 选 |
|---|---|
| 一类多段独立录音 | `recording`(默认) |
| 一类一段长录音 + 多 trial | `trial` |
| 已经预切的 CSV(带 `__group__`)| 当前选什么都行,看 CSV 里 `__group__` 长什么样 |

### 9.4 暖启动(warm-start)

加载前一个 checkpoint 的特征提取器权重,fine-tune 新分类头。**适用场景**:数据持续累积、类别偶尔变化。**坑**:上一个模型如果是泄漏训出来的,暖启动会带过来。

### 9.5 类别加权

类别极度不平衡时(比如 `100:5:100:5:100`),loss 自动按 inverse frequency 加权,防止模型只学多数类。305 平衡数据集开不开都一样。

### 9.6 自动优化(Smart Auto 内部)

3 件事:

1. **架构选择**:根据样本数 / 通道数 / 信号长度自动选 kernel sizes 和 channel widths
2. **类别加权**(如果开了)
3. **早停**:val_loss 连续 N 个 epoch 不下降就停

---

## 10. 数据采集建议

### 10.1 单 session 的天花板

无论你多么完美地走 Prep / Train,**单受试者 + 单 session** 的数字**不能用作泛化证据**——因为模型可能只是学会了「这次贴电极的阻抗模式」。

### 10.2 推荐协议:多 session 多受试者

| 维度 | 数量 |
|------|------|
| **受试者** | ≥ 5 人 |
| **Session / 受试者** | ≥ 4(不同日,每次重新贴电极) |
| **每 session 时长** | 5 min 即可(60 trial / 类) |

### 10.3 文件命名规范

```
data.zip
├── LIU-DAY1-CLENCH.txt
├── LIU-DAY1-FIVE.txt
├── LIU-DAY1-OK.txt
├── LIU-DAY1-ROCK.txt
├── LIU-DAY1-TWO.txt
├── LIU-DAY2-CLENCH.txt   ← 不同日重新贴电极
├── ...
├── WANG-DAY1-CLENCH.txt  ← 不同受试者
├── ...
```

Mode C 的 label map 全部映射到同一个类:

```
LIU-DAY1-CLENCH.txt → CLENCH
LIU-DAY2-CLENCH.txt → CLENCH
LIU-DAY3-CLENCH.txt → CLENCH
WANG-DAY1-CLENCH.txt → CLENCH
...
```

### 10.4 切分粒度改回 recording

多 session 后,**`group_by` 选回 `recording`(默认)**——这时每段录音才是真正独立的"试次",GroupShuffleSplit 会按 session 切,test 集会落到「整段没见过的 session」上。**这个数字才能写进论文。**

### 10.5 Subject-level / Session-level Leave-One-Out

最严格的评估:LeaveOneSubjectOut(把某个受试者的所有数据全留 test)。当前版本还没原生支持,但你可以**手动多次训练 + 切换 test subject**,然后报 mean ± std。这是 v0.2 路线图。

---

## 11. 常见问题与故障排查

### Q1:训练完混淆矩阵全是 100%

**99% 是数据泄漏**。检查:

1. 看混淆矩阵上方有没有 **Evaluation set: held-out test set** 字样,没有 = 泄漏路径
2. 看黄色 banner 是不是触发了
3. 数据是不是直接传了切好的 CSV(没走 Prep)
4. 单受试者单 session 数据,t-SNE 簇分得过分干净 → 即便走了 Prep 也只是「单 session 内的好分」,跨 session 上估计立刻掉到 60%

详见 [R&D-REPORT-2026-04-27.md 附录 A](R&D-REPORT-2026-04-27.md)。

### Q2:训练曲线学不动(loss 不下降)

按概率从高到低排查:

1. **通道数搞错**:列名前缀不是 `ch{N}_M` 但你设了通道数 > 1 → reshape 错了
2. **数据未归一化**(BioSpark-Light 自动做 per-sample normalization,但如果你的信号单位是 μV ~ 1e5,可能数值过大)
3. **类标签拼错**:5 个文件里有 1 个标签写错了,模型学不出
4. **Stride 设得过大**:每类只有 1-2 段,数据量不够

### Q3:进度条不动 / WebSocket 断连

可能是:

- **PyInstaller 打包版**:某些 antivirus 在拦,看托盘里有没有杀毒拦截提示
- **源码版**:看终端有没有 traceback
- **任意版本**:看 `%LOCALAPPDATA%\BioSpark-Light\logs\` 里的最新日志

### Q4:浏览器看不到任何 banner

刷新一下浏览器(Ctrl-F5 强刷)。如果是从源码改完代码,可能 `frontend/js/trainer.js` 缓存了。

### Q5:OpenBCI .txt 上传报错 "Failed to read"

看具体报错。常见:

- 文件其实是 `.gz` 压缩的(OpenBCI GUI 偶尔会压缩) → 解压再传
- 文件被 Excel 重新保存过 → 编码变了(GBK / UTF-8 BOM),导出时选 UTF-8
- 文件**只有 `%`-头没有数据行** → 录的时候没接通设备

### Q6:训完想换数据集再训一轮

直接回 Data Prep 上传新 zip。旧的训练记录在 My Models 里保留,**不会丢**。新训练也不会自动暖启动旧模型(除非你勾)。

### Q7:每段长度不是 1s/5s 怎么办

按你的协议改 Segment Length 和 Stride。例如「每 3s 一段,前 2s 有效」就是 `segment=2, stride=3`。**约束**:`stride ≥ segment`(否则用 overlap_ratio 表达)。

### Q8:任务里没有「休息期」,信号是连续的呢?

把 Stride 留空,改填 Overlap Ratio(0–0.9)。`group_by` 一般还是建议 `recording`(每个文件一组)。

### Q9:不同 OpenBCI 通道数

`Multi-channel Columns` 字段直接改:

| 设备 | 填值 |
|------|------|
| Cyton 4 通道 | `1-4` |
| Cyton 8 通道 | `1-8` |
| Cyton + Daisy 16 通道 | `1-16` |
| 非连续(只用奇数通道) | `1,3,5,7` |

### Q10:可以中文标签吗

可以。`握拳` / `比五` / `OK` / `摇滚` / `比二` 全部可用,训练 / 混淆矩阵 / t-SNE 都会显示中文。

---

## 12. 文档地图

| 文档 | 说明 |
|------|------|
| [README.md](../README.md) | 项目概览、安装快照 |
| [USER-MANUAL.md](USER-MANUAL.md) | **本文档** — 完整用户手册(中文) |
| [USER-MANUAL.en.md](USER-MANUAL.en.md) | Full user manual (English) — same content as this file |
| [USAGE-OPENBCI.md](USAGE-OPENBCI.md) | OpenBCI 数据专项指南(本手册 §8 的扩展版) |
| [BUILD.md](BUILD.md) | PyInstaller 打包流程,开发者用 |
| [R&D-REPORT-2026-04-27.md](R&D-REPORT-2026-04-27.md) | 项目研发报告(架构决策、数据泄漏修复内史) |
| [PROJECT-STATUS-2026-04-27.md](PROJECT-STATUS-2026-04-27.md) | 项目状态快照 |

### 想了解某个具体话题去哪查

| 我想知道... | 查 |
|------------|-----|
| 怎么打包 .exe | [BUILD.md](BUILD.md) |
| 数据泄漏修复的来龙去脉 | [R&D-REPORT 附录 A](R&D-REPORT-2026-04-27.md) |
| 单个字段的取值范围 | 本手册 §4 / §5 的字段表 |
| 我的 OpenBCI 数据具体怎么填 | 本手册 §8 + [USAGE-OPENBCI.md](USAGE-OPENBCI.md) |
| 我的训练效果怎么算诚实 | 本手册 §6 + §10 |

---

*BioSpark-Light · v0.2.0 · 用户手册 · v0.2 更新于 2026-05-03*
*本手册由 xjhveteran199-bit 与 Claude Opus 4.7 结对编写;真实使用案例(§8)的数据来自受试者 LIU 的 5 手势 OpenBCI Cyton 采集。*
