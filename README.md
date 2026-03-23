## vessel-mask (TopCoW runner)

本项目把 TopCoW 2024 官方推理仓库 vendoring 到本地，并提供一个稳定的单文件 CLI：

```bash
uv run vessel-mask-topcow --input /path/case.nii.gz --output /path/case_vessel_mask.nii.gz
```

CLI 会调用上游 `segment_circle_of_willis.py` 做多分类分割，再把非零标签统一转成二值血管 mask。

## 目录约定

```text
vessel-mask/
├── src/vessel_mask/
├── scripts/
│   └── download_topcow_models.py
├── checkpoints/
│   ├── yolo-cow-detection.pt
│   └── topcow-claim-models/
└── third_party/
    └── TopCoW_2024_MRA_winning_solution/
```

## 环境安装

```bash
uv python install 3.11
uv venv --python 3.11
source .venv/bin/activate
uv lock
uv sync
```

`nnunetv2` 已通过 `pyproject.toml` 的本地 editable source 指向：

`third_party/TopCoW_2024_MRA_winning_solution/topcow-2024-nnunet`

## 下载模型

TopCoW 推理需要两类模型：

- YOLO 检测权重：`yolo-cow-detection.pt`
- nnUNet 分割模型目录：`topcow-claim-models/`

本项目默认放在 `checkpoints/`，可用脚本自动下载并解压：

```bash
uv run python scripts/download_topcow_models.py
```

下载时会显示 `tqdm` 实时进度（速度/ETA），并在完成后自动校验文件大小。

可选：

```bash
uv run python scripts/download_topcow_models.py --dry-run
uv run python scripts/download_topcow_models.py --force
```

参考：<https://zenodo.org/records/14191592>

## 环境变量（python-dotenv）

在项目根目录创建 `.env`（可从 `.env.example` 复制）：

```bash
cp .env.example .env
```

关键变量：

- `VESSEL_MASK_DEVICE`：`auto` / `cuda:0` / `mps` / `cpu`
- `VESSEL_MASK_CHECKPOINTS_DIR`：默认 `checkpoints`
- `VESSEL_MASK_YOLO_PATH`：默认 `checkpoints/yolo-cow-detection.pt`
- `VESSEL_MASK_SEGMODEL_DIR`：默认 `checkpoints/topcow-claim-models`
- `VESSEL_MASK_REPO_DIR`：默认 `third_party/TopCoW_2024_MRA_winning_solution`

## 运行推理

```bash
uv run vessel-mask-topcow \
  --input data/input/case001_cta.nii.gz \
  --output data/output/case001_vessel_mask.nii.gz \
  --device auto
```

可选参数：

- `--repo-dir`：TopCoW 仓库目录（默认 `third_party/TopCoW_2024_MRA_winning_solution`）
- `--yolo-path`：YOLO 权重路径（默认 `checkpoints/yolo-cow-detection.pt`）
- `--segmodel-dir`：nnUNet 模型目录（默认 `checkpoints/topcow-claim-models`）
- `--device`：`auto` / `cuda:0` / `cpu` / `mps`
- `--env-file`：指定 dotenv 文件路径

## 重要说明

- 上游脚本默认是 CUDA 入口，本项目会在运行时动态打补丁，不直接改动上游源码。
- Mac 上建议先做流程联调，第一次完整正式推理优先在 Linux + NVIDIA CUDA 环境执行。
