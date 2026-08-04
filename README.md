# GENERATIONIMAGE

面向织物数码喷印的工艺先验、多尺度、纹理保持缺陷生成工程。项目以 O2MAG 为异常外观迁移后端，并增加：

1. 工艺先验 mask：堵头线、banding、白点/飞墨、水渍/墨污；
2. 尺度自适应 ROI：将微小缺陷放大到扩散模型可感知的尺度；
3. 小波高低频融合：保留复杂花型与织物组织；
4. manifest 驱动的数据、参数与消融实验管理。

## Linux 安装

```bash
conda create -n generationimage python=3.10 -y
conda activate generationimage
pip install -e '.[o2mag,eval,dev]'
```

下载 Stable Diffusion 1.5 权重后，将 `configs/fabric_base.yaml` 中的 `model_path` 改为本地目录。代码不会自动下载模型，适合离线服务器。

## 数据清单

输入是 JSONL，每行一个样本：

```json
{"sample_id":"p001_0001","pattern_id":"p001","normal_path":"/data/normal/1.png","reference_path":"/data/real_defect/stain/1.png","reference_mask_path":"/data/real_defect/stain/1_mask.png","defect_type":"stain","split":"train","seed":2026}
```

`nozzle_line`、`banding`、`white_spot` 可不提供参考图；`stain`、`ink_smear` 的 O2MAG 路径需要参考图和参考 mask。

## 运行

先验证无需模型的程序化路径：

```bash
fabric-generate --config configs/fabric_base.yaml --manifest data/train.jsonl \
  --output runs/procedural --routes procedural
```

运行完整方法：

```bash
fabric-generate --config configs/fabric_base.yaml --manifest data/train.jsonl \
  --output runs/full
```

生成质量评价：

```bash
fabric-evaluate --manifest runs/full/manifest.jsonl --output runs/full/metrics.json
```

按花型划分数据，避免同花型泄漏：

```bash
fabric-split --manifest data/all.jsonl --output-dir data/splits --seed 2026
```

## 输出

```text
runs/full/
├── images/<sample_id>.png
├── masks/<sample_id>.png
├── originals/<sample_id>.png
├── metadata/<sample_id>.json
├── manifest.jsonl
└── config.yaml
```

## 消融

- `configs/ablation_o2mag.yaml`：原始 O2MAG，不启用 ROI 和频率融合；
- `configs/ablation_roi.yaml`：启用尺度自适应 ROI；
- `configs/ablation_frequency.yaml`：启用频率融合；
- `configs/fabric_base.yaml`：完整方法。

## O2MAG 来源

`triag/` 保留自 O2MAG 的核心实现。原项目论文：*One-to-More: High-Fidelity Training-Free Anomaly Generation with Attention Control* (2026), arXiv:2603.18093。使用本工程发表成果时请同时引用原论文。

