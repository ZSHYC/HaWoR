# 模型与 MANO

在仓库根目录准备以下文件。公开权重与配置共约 5.4 GB，未随 Git 仓库分发。

```text
weights/
├── external/
│   ├── detector.pt
│   └── droid.pth
└── hawor/
    ├── model_config.yaml
    └── checkpoints/
        ├── hawor.ckpt
        └── infiller.pt
thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth
_DATA/data/mano/MANO_RIGHT.pkl
_DATA/data_left/mano_left/MANO_LEFT.pkl
```

## 公开权重

| 文件 | 来源 |
| --- | --- |
| detector.pt | [WiLoR / Hugging Face](https://huggingface.co/spaces/rolpotamias/WiLoR/blob/main/pretrained_models/detector.pt) |
| hawor.ckpt、infiller.pt、model_config.yaml | [ThunderVVV/HaWoR / Hugging Face](https://huggingface.co/ThunderVVV/HaWoR/tree/main/hawor) |
| droid.pth | [DROID-SLAM 官方 Google Drive](https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view) |
| metric_depth_vit_large_800k.pth | [Metric3D / Hugging Face](https://huggingface.co/JUGGHM/Metric3D/blob/main/metric_depth_vit_large_800k.pth) |

```bash
mkdir -p weights/external weights/hawor/checkpoints thirdparty/Metric3D/weights

curl -fL --retry 3 -o weights/external/detector.pt \
  https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt
curl -fL --retry 3 -o weights/hawor/checkpoints/hawor.ckpt \
  https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/hawor.ckpt
curl -fL --retry 3 -o weights/hawor/checkpoints/infiller.pt \
  https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/infiller.pt
curl -fL --retry 3 -o weights/hawor/model_config.yaml \
  https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/model_config.yaml
curl -fL --retry 3 -o thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth \
  https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_large_800k.pth
curl -fL --retry 3 -o weights/external/droid.pth \
  'https://drive.usercontent.google.com/download?id=1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh&export=download&confirm=t'

sha256sum -c assets/model-weights.sha256
```

Google Drive 可能返回限额或确认网页；若 DROID 校验失败，请从上表原始链接下载后重新校验，不要将网页当作 `.pth` 使用。

[校验文件](../assets/model-weights.sha256) 记录本次成功运行时的 SHA-256。Hugging Face 大权重还与发布端 LFS 哈希比对，配置文件与 Git blob 比对；DROID 使用上游公开链接并验证 PyTorch ZIP 的 CRC，未找到发布端 SHA-256，因此此处的 DROID 哈希是本次下载的复现记录。

## MANO v1.2

1. 在 [MANO 官网](https://mano.is.tue.mpg.de/) 自行注册、接受许可并下载 `mano_v1_2.zip`。
2. 将压缩包中的 `models/MANO_RIGHT.pkl`、`models/MANO_LEFT.pkl` 放到上面的 `_DATA` 路径。

对于官网的 v1.2 压缩包，可在仓库根目录执行以下命令；修改第一行的文件路径：

```bash
MANO_ARCHIVE=/path/to/mano_v1_2.zip
unzip -t "$MANO_ARCHIVE"
mkdir -p _DATA/data/mano _DATA/data_left/mano_left
unzip -p "$MANO_ARCHIVE" mano_v1_2/models/MANO_RIGHT.pkl > _DATA/data/mano/MANO_RIGHT.pkl
unzip -p "$MANO_ARCHIVE" mano_v1_2/models/MANO_LEFT.pkl > _DATA/data_left/mano_left/MANO_LEFT.pkl
chmod 600 _DATA/data/mano/MANO_RIGHT.pkl _DATA/data_left/mano_left/MANO_LEFT.pkl
```

不要重命名左右手模型，也不要把这两个文件或压缩包提交到公开仓库。MANO 使用其自身的 [许可](https://mano.is.tue.mpg.de/license.html)。
