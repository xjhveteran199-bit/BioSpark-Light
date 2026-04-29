# OpenBCI 数据使用指南 — 以 `l.zip` 为例

> **场景**：用 OpenBCI Cyton 采集 5 个手势,每个手势录 5 分钟,采样率 250 Hz,采集 8 个 EXG 通道但实际只接了前 5 个,采集协议是「**每 5 秒一段,每段前 1 秒为有效任务窗口**」(共 60 段/手势)。
>
> 本指南用一份**真实**的 OpenBCI 导出文件 `l.zip` 完整走一遍 BioSpark-Light 的数据准备 → 训练 → 评估流程,所有数字都跟你直接复现。

---

## 0. 这份数据长什么样

`l.zip` 解压后:

```
l/
  LIU-CLENCH.txt   ← 9.5 MB
  LIU-FIVE.txt     ← 9.4 MB
  LIU-OK.txt       ← 9.5 MB
  LIU-ROCK.txt     ← 9.4 MB
  LIU-TWO.txt      ← 9.5 MB
```

5 个 `.txt` **平铺**在 `l/` 文件夹下,**不是 folder-per-class** 结构。这意味着:

> ⚠️ 必须用 **Mode C(generic ZIP + per-file label)**,不能用 Mode A(folder-per-class)。

每个 .txt 是 OpenBCI GUI 的「无表头」格式:

```
%OpenBCI Raw EEG Data
%
%Sample Rate = 250.0 Hz
%First Column = SampleIndex
%Last Column = Timestamp
%Other Columns = EEG data in microvolts followed by Accel Data (in G) interleaved with Aux Data
0, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.000, 0.000, 0.000, 18:51:16.059
1, -125246.01, -59240.19, 3127.81, -171355.80, -118593.41, -187500.02, -187500.02, -187500.02, 0.000, 0.000, 0.000, 18:51:16.074
2, -124946.98, ...
```

13 列:`[SampleIndex, EXG×8, Accel×3, Timestamp]`。前 5 个 EXG(c1~c5)是真实接通的,后 3 个 EXG(c6/c7/c8)显示 `-187500.02` 表示**未接电极**(ADC 饱和),Accel 三轴(c9/c10/c11)全是 0。

---

## 1. 启动应用

双击 `dist/BioSpark-Light/BioSpark-Light.exe`。看到日志:

```
[biospark.launcher] Server: http://127.0.0.1:8765
INFO: Application startup complete.
```

浏览器自动打开。如果用的是源码:

```
cd D:\Claude code\BioSpark-Light
python launcher.py
```

---

## 2. 走 Prep 标签页(**不要**直接进 Train)

### 2.1 上传

1. 切到 **Data Prep** 标签页
2. 选 **Mode C — generic ZIP**
3. 把 `l.zip` 拖进上传框

上传完成后,后台会做 inspect,自动识别 OpenBCI 头部并填好下面这些字段:

| 字段 | 自动填值 | 解释 |
|------|---------|------|
| Sampling Rate (Hz) | **250** | 从 `%Sample Rate = 250.0 Hz` 读出 |
| Multi-channel Columns | **`1-5`** | OpenBCI 8 通道里只有前 5 个真实采集,后 3 个饱和会被自动跳过;col 0(SampleIndex)和 col 9–12(Accel/Timestamp)也会跳过 |
| Signal Column Index | 0 | 旧版单列字段,**多通道字段填了就忽略它** |

### 2.2 手动设的 4 个关键字段

| 字段 | 填值 | 为什么 |
|------|------|------|
| **Segment Length (sec)** | `1` | 每段前 1 秒是有效任务窗口 |
| **Stride (sec, optional)** | `5` | 每 5 秒一段,跳过中间 4 秒休息期 |
| **Overlap Ratio** | `0` | 设了 stride 就别再用 overlap |
| **Group-aware split unit** | **`Trial (each segment = 1 trial)`** | 一个手势只有 1 段长录音 + 60 个独立 trial,必须选 Trial 级,否则一整个 class 会被丢到 test 集 |

### 2.3 文件 → 类别映射(Mode C 必填)

下方的 Per-file label 表里把 5 行填上(可以复制粘贴):

| File | Class label |
|------|-------------|
| `l/LIU-CLENCH.txt` | `CLENCH` |
| `l/LIU-FIVE.txt`   | `FIVE` |
| `l/LIU-OK.txt`     | `OK` |
| `l/LIU-ROCK.txt`   | `ROCK` |
| `l/LIU-TWO.txt`    | `TWO` |

类标签想用中文(`握拳` / `比五` / `OK` / `摇滚` / `比二`)也行,会原样进训练。

### 2.4 点 Generate Training CSV

成功后会显示:

```
Generated 305 samples across 5 classes.
```

每类 61 段(5min 多录了一点点 → `(75767 - 250) / 1250 ≈ 60.4` → 61),签出参数:

| 维度 | 值 |
|------|-----|
| 样本总数 | 305 |
| 每类样本 | CLENCH 61 / FIVE 61 / OK 61 / ROCK 61 / TWO 61 |
| 信号长度 | 250(=1s × 250Hz) |
| 通道数 | 5(自动检测) |
| `__group__` 列 | 305 个唯一 trial id |
| 列总数 | 1252(`ch1_1..ch5_250` + `label` + `__group__`) |

点 **Use for Training** 把这份准备好的数据集直接送到 Train 标签页(也可以 Download 下载 CSV 留底)。

---

## 3. 训练

切到 **Train** 标签页,数据集已经自动载入。

### 3.1 检查数据摘要

如果刚才 Prep 跑成功了,数据预览下方**不应该**出现黄色「This dataset has no `__group__` column」banner。如果出现了,说明你跳过了 Prep 直接传了切好的 CSV,回到 §2 重做。

### 3.2 配置

| Preset | 推荐 | 备注 |
|--------|------|------|
| **Smart Auto** | ✅ 默认 | 自动选架构 + 类别加权 + 早停;不开 LR 搜索;~3–10 分钟 |
| Quick Test | 想快看一眼时 | 20 epochs,~1–2 分钟 |
| Publication Ready | 投稿前最后一跑 | 100 epochs + LR 搜索;15–45 分钟 |

把 **Number of channels** 留空(=auto),它会从列名前缀 `ch{N}_{i}` 自动检测出 5 通道。

**Warm-start** 第一次训练保持不勾。

### 3.3 启动 + 看 split 模式

点 **Start Training**。Epoch 日志的**第一行**会显示切分模式:

```
Split: 逐行随机切分（缺少 __group__ 列） · train=180 val=60 test=60   ← ❌ 错误,Prep 没跑通
Split: 组级切分 · 305 个录音组 · train=183 val=61 test=61              ← ✅ 正确,group_by=trial
```

颜色:绿色 = 组级,黄色 = 逐行随机。

### 3.4 训练曲线

Loss / Accuracy 实时画在右上角。一般 30–50 epoch 早停。

---

## 4. 看结果(关键的来了)

### 4.1 混淆矩阵

训练完成后跳到 **T5. 训练结果可视化**。混淆矩阵的左上方会标:

```
Evaluation set: held-out test set
```

而不是「validation set (no held-out test)」。

### 4.2 黄色「疑似数据泄漏」banner

如果你看到这个 banner,**不要忽略**:

> ⚠️ Suspiciously high accuracy — possible data leakage
>
> Validation accuracy is ≥99% on a small set, and there is no held-out test set...

什么时候会触发?这三个条件**全部**满足时:

1. 准确率 ≥ 99%
2. 任一类的支持数 < 30
3. **要么**没走 Prep(group_aware=False),**要么**测试集太小(<30/类)

对你这份数据(每类 61 段,test 集 ≈12 段/类),**第 2 条仍会满足** → banner 仍会 fire,即便切分是正确的。这是诚实的提示:**12 个样本测出 100% 不能当真**。要消掉这个 banner,需要每类至少 ~150 个独立 trial,也就是 **多个 session** 各录几段。

### 4.3 t-SNE

5 个簇分得有多干净?干净不一定是好事——可能是:

(a) 任务**真的好分**(单受试者、5 个差别巨大的手势、电极位置一致)
(b) 同一段 5min 录音里**相邻 trial 之间的电极漂移**让模型记住了"这是录音前段还是后段"

`group_by=trial` **不能区分这两种情况**。

---

## 5. 数据足够发表吗?

**不够**。即使你这次拿到了 95%+ 的 test 准确率,这个数字不能写进论文 / 报告 — 原因:

1. **单受试者**(Subject = LIU)
2. **单 session**(每个手势只录 1 次,所有 trial 在同 1 段录音里)
3. **测试集太小**(每类 ~12 个 trial,统计噪声大)

要做能发的实验,推荐采集协议:

| 维度 | 现状 | 推荐 |
|------|------|------|
| 受试者 | 1 人 | **≥ 5 人** |
| Session 数 | 1 session/类 | **≥ 4 session/受试者**(不同日,每次重新贴电极) |
| 单 session 长度 | 5 min | 保持 5 min(60 trial)即可 |
| 切分方式 | trial 级随机 | **subject 级 LeaveOneSubjectOut** 或 **session 级 LeaveOneSessionOut** |

每多一个 session,你都要按 `LIU-DAY1-CLENCH.txt / LIU-DAY2-CLENCH.txt / ...` 命名,Mode C 的 label 映射里都映射到同一个 `CLENCH` 类。这样 Prep 输出的 `__group__` 自然以文件名为单位,组级切分就能把整段录音独立分到 train 或 test,泄漏 banner 也会自然消失。

---

## 6. 一行 Python 复现(命令行用户)

如果你不想点 UI,等价的 Python 调用是:

```python
from backend.services.data_preparator import segment_generic
from backend.services.dataset_loader import load_labeled_dataframe

with open('C:/Users/XJH-V/Desktop/l.zip', 'rb') as f:
    zip_bytes = f.read()

df = segment_generic(
    zip_bytes,
    file_label_map={
        'l/LIU-CLENCH.txt': 'CLENCH',
        'l/LIU-FIVE.txt':   'FIVE',
        'l/LIU-OK.txt':     'OK',
        'l/LIU-ROCK.txt':   'ROCK',
        'l/LIU-TWO.txt':    'TWO',
    },
    signal_col_index=0,
    signal_col_indices=[1, 2, 3, 4, 5],
    sampling_rate=250.0,
    segment_length_sec=1.0,
    overlap_ratio=0.0,
    stride_sec=5.0,
    group_by='trial',
)
print(df.shape)             # (305, 1252)
print(df['label'].value_counts())
print(df['__group__'].nunique())   # 305

summary = load_labeled_dataframe(df)
print(summary['n_channels'])      # 5
print(summary['signal_length'])   # 250
```

输出和 UI 流程**完全一致**。

---

## 7. 常见问题

### Q1:我有不同 OpenBCI 通道数(4/8/16),怎么办?

`Multi-channel Columns` 字段直接改成你想要的范围:

- Cyton 4 通道:`1-4`
- Cyton 8 通道全用:`1-8`
- Cyton + Daisy 16 通道:`1-16`

非连续也行,例如 `1,3,5,7`。

### Q2:OpenBCI 文件没有 `%` 头怎么办?

reader 也接受**无注释、无表头**的纯数字 CSV/TXT,直接传就行——会自动给列起名 `c0..cN`。

### Q3:每段长度不是 1s/5s 怎么办?

按你的协议改 **Segment Length** 和 **Stride**。例如「每 3s 一段,前 2s 有效」就是 `segment=2, stride=3`。

### Q4:任务里没有「休息期」,信号是连续的呢?

把 **Stride** 留空,改填 **Overlap Ratio**(0–0.9)即可走传统重叠切分。`group_by` 仍然建议选 Trial,除非每个 trial 都是一个独立 .txt 文件。

### Q5:训练完了想再换个手势集合?

回到 **Data Prep** 重新上传新 zip 即可,旧的训练记录在 **My Models** 里保留(每次训练都会落库 + 存 checkpoint,不会丢)。

---

*本指南配合 [R&D-REPORT-2026-04-27.md](R&D-REPORT-2026-04-27.md) 的「附录 A — v0.1.1 修复轮次」一起看,后者解释了为什么必须做组级切分、为什么 trial 级 grouping 不能等同于跨 session 泛化。*
