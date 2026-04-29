# BioSpark-Light 研发报告

> **目的**：项目级综合快照——已完成的功能、关键技术决策、代码资产、未来 1-3 个月路线图。
> **日期**：2026-04-27
> **版本**：v0.1.0-light
> **仓库**：<https://github.com/xjhveteran199-bit/BioSpark-Light>

---

## 1. 项目定位

BioSpark-Light 是 [BioSpark](https://github.com/) 的**离线桌面分支**——同款 1D-CNN 训练引擎、同款出版级图表，但删去云端必需的部分（用户认证、推理 API、实时流监控），改为本机 `python launcher.py` 或双击 `.exe` 即可运行的桌面应用。

**核心价值主张**：研究者不希望把医疗 / 生理数据上传到服务器。Light 在本地 CPU 上跑同样的训练流水线，所有数据写入用户家目录的 per-user 数据夹。分享 `.pt` 模型而不分享原始数据。

**目标用户**：科研人员、临床实验员、任何手里有生物信号 CSV/ZIP 想训分类器但不想配 Python 环境的人。

---

## 2. 已完成 ✅

### 2.1 三种数据整理模式（Data Prep）

`backend/services/data_preparator.py` (382 行) + `backend/routers/prep.py` (207 行) + `frontend/js/prep.js` (353 行)

| 模式 | 输入 | 用途 |
|------|------|------|
| **Mode A** | folder-per-class ZIP，每个 CSV 是一段长录制 | 最常见的实验数据组织方式 |
| **Mode B** | 单个长 CSV + label 时间区间表 | 一次连续记录后人工标注的数据 |
| **Mode C** | 通用 ZIP，UI 表格手工映射 file→label→signal_col | 兜底方案，命名/格式不规整也能用 |

三个 segmenter 共享 `preprocess._segment` 滑窗原语（采样率 / 段长 / overlap 三参数化），输出统一为训练器可消费的 CSV，可一键 `promote` 到训练池。

### 2.2 训练流水线

`backend/services/trainer.py` (714 行) + `backend/services/auto_optimizer.py` (385 行) + `backend/routers/training.py` (663 行)

- **Signal1DCNN** 架构（卷积 + BN + ReLU + Pool 堆叠 + 自适应池 + FC 头），通道数 / 段长自适应
- **Auto-Optimizer**：可选 LR Range Test、Early Stopping、自动类别权重平衡、按数据规模选 batch / epoch
- **Warm-Start 续训**：能从该用户最佳 checkpoint 加载 features 部分，分类头 shape 不匹配时自动 reinit（partial warm-start），WS 上明确告知
- **WebSocket 实时进度**：epoch / loss / acc / lr / lr_search_progress 全部推送到前端
- **三种预设**：Quick Test (30s-2min) / Smart Auto (3-10min) / Publication Ready (15-45min)

### 2.3 模型版本管理（My Models）

`backend/models/training_history.py` (85 行) + `backend/routers/model_history.py` (144 行) + `frontend/js/my_models.js` (152 行)

- 每用户私有版本链（`v1` → `v2` → … `is_active=True` 唯一）
- Checkpoint 落盘格式自包含 `state_dict + n_classes + class_names + input_shape + arch_config`，离线即可校验兼容性
- UI 提供 Activate / Delete / 准确率折线图，能直观看出「v3 比 v2 提升了 5%」

### 2.4 出版级图表（T6）

`backend/services/publication_figures.py` (707 行) + `backend/routers/figures.py` (323 行) + `frontend/js/figures.js` (133 行)

- **5 张图**：训练曲线、混淆矩阵、t-SNE、各类指标、模型架构
- **3 种期刊样式**：Nature / IEEE / Science（字体、配色、宽度按各刊规范）
- **2 种格式**：PNG (300 DPI) + SVG 矢量图
- **bulk ZIP 下载**：5 张图 × 2 格式并发渲染，180s timeout，失败的单张写为 `<name>.failed.txt` 而非整体 500
- 复用父项目 `publication_figures.py`（v0.1 漏拷已修复，commit `4a9b533`）

### 2.5 桌面集成

`launcher.py` (161 行)

- 嵌入 uvicorn 在后台线程跑
- 启动后自动开浏览器到 `http://127.0.0.1:8765`
- 系统托盘图标（pystray + PIL），右键菜单 Quit / Open Data Folder
- CLI 选项 `--no-tray --no-open --port` 方便开发与服务器场景
- **数据严格分离**：只读资源在 bundle 里，可写数据在 `platformdirs.user_data_dir(APP_NAME, appauthor=False)`
  - Windows: `%LOCALAPPDATA%\BioSpark-Light\`
  - macOS: `~/Library/Application Support/BioSpark-Light/`
  - Linux: `~/.local/share/BioSpark-Light/`

### 2.6 打包与发布

`biospark-light.spec` (132 行) + `docs/BUILD.md` (90 行)

- **PyInstaller 一文件夹模式**（非 onefile，因为 torch DLL 在 onefile 解压模式下不稳定）
- `collect_all` 抓全 torch / sklearn / scipy / matplotlib
- 显式 25 项 `extra_hidden`，14 项 `excludes` 切掉 tkinter / Qt / Jupyter / pytest / torchvision / torchaudio / mne 等用不到的大块头
- **Bundle 体积**：747 MB（CPU-only torch wheel；CUDA wheel 会涨到 ~3.5 GB）
- **构建时间**：~6 min 30 s
- **GitHub 仓库**：已公开发布，3 个 commits（初始 / 打包 / figures hotfix）

---

## 3. 代码资产（按规模）

| 区域 | 行数 | 文件数 | 说明 |
|------|------|--------|------|
| `backend/services/` | 2 760 | 7 | trainer / publication_figures / auto_optimizer / data_preparator / preprocess / dataset_loader / dataset_cache |
| `backend/routers/` | 1 337 | 4 | training / figures / prep / model_history |
| `backend/` 核心 | 281 | 4 | main / config / database / models |
| `frontend/css/` | 2 057 | 1 | 单文件设计系统，含暗色主题 |
| `frontend/js/` | 1 687 | 5 | trainer / prep / my_models / figures / app |
| `frontend/index.html` | 468 | 1 | 单页三 tab + T6 模态 |
| `launcher.py` + `*.spec` | 293 | 2 | 桌面入口 + 打包 spec |
| `docs/` | 292 | 2 | BUILD.md + 上一份 R&D 报告 |
| **总计** | **≈ 9 175** | **26** | 净写代码（不含 dist / build / venv） |

**特别说明**：项目代码里 `TODO / FIXME / XXX` 计数为 **0**——意味着没有积压的「以后再说」标记，所有已知问题要么已修，要么进了下面的「未来计划」。

---

## 4. 关键技术决策（与备选）

| 决策 | 备选 | 选定理由 |
|------|------|----------|
| PyInstaller **one-folder** | one-file / Briefcase / Nuitka | torch DLL 在 onefile 解压不稳；one-folder 冷启动 2-3s vs onefile 10-30s；杀软误报率低 |
| **CPU-only torch wheel** | 默认 CUDA wheel | bundle 从 ~3.5 GB → 747 MB；本地训练 inference-free，CPU 够用 |
| **platformdirs** 分离 user data | 把 DB 放 .exe 旁边 | 安装到 Program Files 也可写；卸载/重装不丢数据；多用户隔离 |
| `sys.frozen` 检测 | 双套代码 | `backend/config.py` 一处分支，源码模式与 frozen 模式共用同一份代码 |
| `console=True`（v0.1） | 直接 windowed | 早期用户少、问题多；启动黑屏 = 排错地狱；UI 稳定后改 False |
| 不开 UPX | 开 UPX 压到 ~450 MB | UPX 经常破坏 numpy/torch native DLL，稳定优先 |
| **vanilla JS**，无 build step | React/Vue/Svelte | 桌面应用迭代节奏，无 build 让 hack-then-refresh 链路最短 |
| **SQLite + aiosqlite** | Postgres / 文件 JSON | 单机零配置；async 与 FastAPI 一致；将来要多用户再上 PG |

---

## 5. Bug 修复历史

| 日期 | Bug | 根因 | 修复 |
|------|-----|------|------|
| 2026-04-27 | T6 图表全部 HTTP 500，ZIP 下载失败 | `backend/services/publication_figures.py` 在初始从 BioSpark 复刻时漏拷；`figures.py:40` 懒加载它，所以源码与 .exe 都炸 | commit `4a9b533`：从父项目复制 707 行模块 + spec hidden imports 加入；重建 bundle 验证 |
| (历史) | `/api/train/{job_id}/figures/all.zip` 串行渲染易超时 + 单图失败带崩整包 | 同步 await 5×2 渲染；无 timeout；无 partial-failure 容错 | 改为并发 + 180s timeout + 失败写 `.failed.txt` 入 zip + Content-Length header |

---

## 6. 未来计划

### 6.1 短期（接下来 1-2 周）

| 优先级 | 项 | 备注 |
|--------|----|------|
| **P0** | T6 图表 **端到端实测** | 拿真实训练完成的 job 跑一遍 5 图 ZIP 下载，确认像素级正确（目前只验证了 import + 404 路径，没验证渲染结果） |
| **P0** | GitHub **Release v0.1.0** + zip 上传 | 把 `dist/BioSpark-Light/` 打 zip，挂 GitHub Releases 页，让用户能下载，不用 git clone |
| **P1** | **Windows 代码签名** 调研 | 当前杀软误报率高；DigiCert / Sectigo 证书 ~$200/年；评估 ROI |
| **P1** | `console=False` **windowed 模式** | 跑过几次端到端验证后切，纯托盘体验更专业 |
| **P2** | **macOS / Linux 构建** | spec 跨平台兼容；需要 macOS / Linux 机器或 GitHub Actions runner |

### 6.2 中期（1-2 月）

| 项 | 说明 |
|----|------|
| **Auto-update 通道** | manifest JSON 放 GitHub；启动期 check + 后台下载新 zip + 提示重启；考虑用 `pyupdater` 或自写 |
| **GitHub Actions CI** | push 到 main 触发：lint → smoke test → 三平台 PyInstaller 构建 → 上传到 Releases |
| **License-gated** 高级特性 | 基础免费，高级（如 GradCAM、ONNX 导出、批量推理）需 license key；用对称密钥 + offline activation 避免服务器依赖 |
| **国际化** | 已有中英双语切换骨架，扩展到 ja / es / fr 等；图表标签也需国际化 |
| **测试覆盖** | 当前没有测试套件；至少为 `data_preparator` / `trainer` / `figures` 加 happy-path pytest |

### 6.3 长期（3 月以上）

| 项 | 说明 |
|----|------|
| **桌面壳替换** | 当前是浏览器 + 托盘；考虑 Tauri (Rust+WebView) 或 PyWebView 提供更原生的窗口体验。优势：去掉浏览器陷阱（用户关浏览器误以为应用退了）；劣势：要改打包流程 |
| **插件系统** | 让用户提交自定义模型架构（如 Transformer、ResNet1D）；定义 plugin API（必须实现 `Model.forward / num_params / arch_summary`） |
| **联邦学习探索** | 多研究所协作训练同一个模型，但数据不出本地；技术上是把 warm-start chain 扩成「跨用户合并梯度」，需要密码学协议；属于研究性 feature |
| **GPU 选项** | 检测到 CUDA 时让用户选；目前 CPU-only |
| **导出 ONNX** | 训练完成后一键导出，方便嵌入到 C++ / Edge 设备 |

### 6.4 不会做的事（明确划界）

- **回归云端推理 API** — Light 的核心定位就是离线，不破坏
- **多用户 Web 部署** — 那是父项目 BioSpark 的领地
- **真实时数据流监控** — 同上，留给父项目

---

## 7. 风险与依赖

- **PyInstaller × torch 兼容性**：每次 torch 大版本更新都可能破坏 hidden imports；需要在 CI 跑 smoke test 接住
- **杀软误报**：未签名的 PyInstaller .exe 在 Windows Defender / 360 / 火绒下大概率被拦；签名是必需，不是 nice-to-have
- **bundle 体积持续增长**：每加一个新依赖都可能涨 50-200 MB；spec 的 `excludes` 列表要持续维护
- **Python 3.14 + torch / pydantic**：当前 Python 3.14 跑 pydantic v1 有 warning（pydantic v2 OK）；torch wheels 在 3.14 还没全 GA，构建时偶有兼容性问题

---

## 8. 复现步骤（开发者上手）

```bash
git clone https://github.com/xjhveteran199-bit/BioSpark-Light.git
cd BioSpark-Light
python -m pip install pyinstaller platformdirs pystray pillow
python -m pip install -r requirements.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# 源码运行
python launcher.py

# 打包
python -m PyInstaller --noconfirm --clean biospark-light.spec
./dist/BioSpark-Light/BioSpark-Light.exe
```

---

*本报告由 xjhveteran199-bit 与 Claude Opus 4.7 结对汇总。如需更新（添加新 bug 修复、调整路线图优先级），直接编辑此文件即可。*
