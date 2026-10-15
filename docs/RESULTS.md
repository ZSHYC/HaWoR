# HaWoR 复现与多视频推理记录

日期：2026-10-15。**完成4段视频的定性推理与可视化，其中1段来自独立外部数据。没有进行有真值的精度评测。**

![四段相机叠加预览](../assets/results/preview.gif)

GIF仅展示各输入前4秒，6 fps；MP4保留30 fps与完整导出帧。左手紫色，右手蓝色。

- [本地交互式报告](results/index.html)：克隆仓库后可在浏览器打开，切换案例、视频与图表；GitHub文件页不会直接执行HTML。
- [完整结果下载](https://github.com/ZSHYC/HaWoR/releases/tag/reproduction-2026-10-15)：包含1080p相机/世界导出与离线报告。
- [机器可读实验清单](../assets/results/experiments.json)：输入哈希、来源、帧数、耗时及工具版本。

## 环境与执行范围

Ubuntu22.04，单任务使用1张NVIDIA A80080GB；Python3.10.22、PyTorch1.13.0+cu117、CUDA11.7。安装配置与公开权重校验见[安装指南](INSTALL.md)、[模型准备](MODELS.md)。推理代码基于`41d4cb9`，没有改模型、检查点或推理算法；此次增加离线诊断与媒体整理工具。

每段均完成检测/跟踪、HaWoR、masked DROID-SLAM、Metric3D尺度估计、运动补全与world/cam离屏导出。`video_0`沿用此前跑通的缓存，其余3段是首次处理该输入。本次额外两个官方视频并行运行在不同GPU上，所有耗时都包括加载、计算、I/O和渲染，不是论文或单模型FPS基准。

所有输入、原始相机/世界导出、三视角与局部对照视频均完整解码检查。上游保留末帧排他切片：N帧预测只导出N−1帧，本报告不补造末帧。

## 数据与结果概览

| 输入 | 来源 | 输入→导出帧 | 输入时长→导出时长 | 首次处理该输入+world | 复用缓存导出cam |
|---|---|---:|---:|---:|---:|
| `video_0` | HaWoR仓库示例 | 121→120 | 4.03s→4.00s | 本轮未计时 | 本轮未计时 |
| `segment_018` | HaWoR仓库示例 | 120→119 | 4.00s→3.97s | 84.9s | 24.1s |
| `segment_037` | HaWoR仓库示例 | 120→119 | 4.00s→3.97s | 86.0s | 23.9s |
| `openaoe_pouch` | Open-AoE外部数据 | 180→179 | 6.00s→5.97s | 120.0s | 27.4s |

四段输入均为1920×1080、30fps。运行耗时不包含此前环境安装/扩展编译/公开权重下载；cam复用同输入的中间结果，不能与首次完整处理耗时当作两种模型速度来比较。

## 覆盖与补全统计

以下均是**缓存支持/预测可用性的帧数**，左右顺序固定为左/右。分母为完整预测长度N，包含未导出的最后一帧。

| 输入 | 原始检测侧别支持 左/右 | 相机阶段支持 左/右 | 最终有效（含补全）左/右 | 最终有效但无相机支持 左/右 |
|---|---:|---:|---:|---:|
| `video_0` | 112/121 · 87/121 | 115/121 · 84/121 | 121/121 · 120/121 | 6 · 36 |
| `segment_018` | 64/120 · 104/120 | 62/120 · 106/120 | 120/120 · 120/120 | 58 · 14 |
| `segment_037` | 95/120 · 120/120 | 95/120 · 120/120 | 120/120 · 120/120 | 25 · 0 |
| `openaoe_pouch` | 180/180 · 180/180 | 180/180 · 180/180 | 180/180 · 180/180 | 0 · 0 |

原始检测统计来自保留的`model_tracks.npy`记录，按当帧原始左右手标签去重；track多数侧别重分配另在JSON/图中记录。相机阶段支持可包含插值框。最终`pred_valid`会被补全更新，**不是检测可靠性、真实可见性或准确率**。例如`segment_018`左手最终120/120帧有效，却只有62/120帧有相机阶段支持。

四段最终预测的trans/rot/hand_pose/betas/valid五组数组全部有限；这是数值检查，不代表姿态正确。原代码的`model_boxes.npy`为空，诊断改读实际存有记录的`model_tracks.npy`，没有改动推理缓存。

## 分案例可视化与观察

三视角视频逐帧对齐原图、相机叠加和上游固定世界视角；手部局部视频使用检测框驱动的同一裁剪窗口处理原图和叠加图，仅改变显示区域。检测缺失时裁剪窗口插值，并在视频内标注。没有修改模型预测。

上游世界观察相机固定且较远，因此手部在世界视频中较小。另提供拟合坐标范围的静态网格三视图，方便看清几何；每例选最接近中点、两侧原始检测和最终预测均有支持的帧，规则与选中帧写入`mesh_view.json`。这不是选帧后的精度评价。

### video_0：官方示例：端盘与转头

转头时手部会离开视野；最终有效输出包含补全。右手最后一个预测帧仍无效，并在诊断中保留。

[三视角视频](../assets/results/video_0/comparison.mp4) · [手部局部视频](../assets/results/video_0/hand_detail.mp4) · [逐帧统计](../assets/results/video_0/framewise.json) · [详细数值与来源](../assets/results/video_0/summary.json)

![video_0关键帧](../assets/results/video_0/keyframes.jpg)

![video_0原始预测网格三视图](../assets/results/video_0/04_mesh_views.png)

<details>
<summary>查看检测支持、平移分量和三维轨迹图</summary>

![检测支持与分数](../assets/results/video_0/01_support_and_detection.png)

![世界与相机平移参数](../assets/results/video_0/02_translation_components.png)

![轨迹与逐帧位移](../assets/results/video_0/03_world_trajectory_and_motion.png)

</details>

### segment_018：额外示例：取调料与开盖

左手相机阶段仅覆盖62/120帧；开头58帧最终有输出但没有相机阶段支持。观察画外补全及指尖贴合偏差。

[三视角视频](../assets/results/segment_018/comparison.mp4) · [手部局部视频](../assets/results/segment_018/hand_detail.mp4) · [逐帧统计](../assets/results/segment_018/framewise.json) · [详细数值与来源](../assets/results/segment_018/summary.json)

![segment_018关键帧](../assets/results/segment_018/keyframes.jpg)

![segment_018原始预测网格三视图](../assets/results/segment_018/04_mesh_views.png)

<details>
<summary>查看检测支持、平移分量和三维轨迹图</summary>

![检测支持与分数](../assets/results/segment_018/01_support_and_detection.png)

![世界与相机平移参数](../assets/results/segment_018/02_translation_components.png)

![轨迹与逐帧位移](../assets/results/segment_018/03_world_trajectory_and_motion.png)

</details>

### segment_037：额外示例：扶锅与搅拌

右手相机阶段覆盖120/120帧；左手在86–110帧缺少相机阶段支持，最终输出由补全延续。

[三视角视频](../assets/results/segment_037/comparison.mp4) · [手部局部视频](../assets/results/segment_037/hand_detail.mp4) · [逐帧统计](../assets/results/segment_037/framewise.json) · [详细数值与来源](../assets/results/segment_037/summary.json)

![segment_037关键帧](../assets/results/segment_037/keyframes.jpg)

![segment_037原始预测网格三视图](../assets/results/segment_037/04_mesh_views.png)

<details>
<summary>查看检测支持、平移分量和三维轨迹图</summary>

![检测支持与分数](../assets/results/segment_037/01_support_and_detection.png)

![世界与相机平移参数](../assets/results/segment_037/02_translation_components.png)

![轨迹与逐帧位移](../assets/results/segment_037/03_world_trajectory_and_motion.png)

</details>

### openaoe_pouch：外部输入：Open-AoE 黑色小包

独立外部数据的两只手均有180/180帧检测支持，但叠加图仍有手掌偏大、指尖贴合不足。满覆盖不等于准确。

[三视角视频](../assets/results/openaoe_pouch/comparison.mp4) · [手部局部视频](../assets/results/openaoe_pouch/hand_detail.mp4) · [逐帧统计](../assets/results/openaoe_pouch/framewise.json) · [详细数值与来源](../assets/results/openaoe_pouch/summary.json)

![openaoe_pouch关键帧](../assets/results/openaoe_pouch/keyframes.jpg)

![openaoe_pouch原始预测网格三视图](../assets/results/openaoe_pouch/04_mesh_views.png)

<details>
<summary>查看检测支持、平移分量和三维轨迹图</summary>

![检测支持与分数](../assets/results/openaoe_pouch/01_support_and_detection.png)

![世界与相机平移参数](../assets/results/openaoe_pouch/02_translation_components.png)

![轨迹与逐帧位移](../assets/results/openaoe_pouch/03_world_trajectory_and_motion.png)

</details>

## 外部数据的来源与限制

外部输入来自[Open-AoE / inclusionAI OpenAoE-2000h](https://huggingface.co/datasets/inclusionAI/OpenAoE-2000h)，对应论文[Open-AoE](https://arxiv.org/abs/2607.14183)。使用原始视频样本`aoe_20260201_192308_p000`的0–6秒；只使用视频和相机元数据，没有使用该数据集内置的手部重建结果作为真值。

固定数据版本：`3fd28613804145b6a0ec7a81ac3dd40918158991`。原始MP4 SHA-256：`41c5bed523e4b3c946d34a58417bd6eeee833a0baa3dbf9c8bf989001376e64b`。公开预览包含该数据的派生画面，保留[Open-AoE许可](../assets/results/licenses/OpenAoE.txt)与署名，不表示数据提供方认可本复现。

元数据给出fx约787.9、fy约591.3；本次统一使用HaWoR默认单焦距600像素、居中主点，未额外去畸变。这些近似以及场景差异可能影响结果，但本次没有进行误差归因。外部叠加中可见手掌偏大、指尖贴合不足；检测覆盖100%不构成泛化精度证明。

## 如何重跑和整理自己的结果

先按安装指南激活环境，在仓库根目录执行：

```bash
conda activate hawor
CUDA_VISIBLE_DEVICES=0 python demo.py --video_path example/segment_018.mp4 --vis_mode world --headless
CUDA_VISIBLE_DEVICES=0 python demo.py --video_path example/segment_018.mp4 --vis_mode cam --headless

python scripts/results/diagnostics.py --sequence example/segment_018 --output /tmp/hawor-results/segment_018/diagnostics
python scripts/results/mesh_views.py --repo . --sequence example/segment_018 --output /tmp/hawor-results/segment_018/diagnostics
python scripts/results/build_media.py \
  --input example/segment_018.mp4 \
  --cam example/segment_018/vis_cam_0_119/aitviewer/video.mp4 \
  --world example/segment_018/vis_world_0_119/aitviewer/video.mp4 \
  --tracks example/segment_018/tracks_0_120/model_tracks.npy \
  --output /tmp/hawor-results/segment_018/media --title segment_018
```

诊断和网格工具只读取自己信任的推理缓存。诊断不运行模型；网格工具需要MANO，执行CPU参数化前向；媒体工具不运行推理，需要FFmpeg。工具支持输出到缓存外的目录，防止整理结果时覆盖输入。

外部输入的准备命令如下，`EXTRA_DIR`可以换成自己的数据目录：

```bash
EXTRA_DIR="$HOME/data/hawor-extra"
mkdir -p "$EXTRA_DIR"
curl -fL --retry 3 -o "$EXTRA_DIR/raw_video.mp4" \
  'https://huggingface.co/datasets/inclusionAI/OpenAoE-2000h/resolve/3fd28613804145b6a0ec7a81ac3dd40918158991/aoe_20260201_192308_p000/raw_video.mp4'
curl -fL --retry 3 -o "$EXTRA_DIR/OpenAoE-LICENSE.txt" \
  'https://huggingface.co/datasets/inclusionAI/OpenAoE-2000h/resolve/3fd28613804145b6a0ec7a81ac3dd40918158991/LICENSE'
# 核对上面记录的原始MP4 SHA-256。
ffmpeg -nostdin -i "$EXTRA_DIR/raw_video.mp4" -t 6 -an -vf fps=30 \
  -c:v libx264 -crf 18 -pix_fmt yuv420p "$EXTRA_DIR/openaoe_pouch.mp4"
CUDA_VISIBLE_DEVICES=0 python demo.py --video_path "$EXTRA_DIR/openaoe_pouch.mp4" --img_focal 600 --vis_mode world --headless
CUDA_VISIBLE_DEVICES=0 python demo.py --video_path "$EXTRA_DIR/openaoe_pouch.mp4" --img_focal 600 --vis_mode cam --headless
```

## 读图边界与后续工作

- 平移曲线是保存的MANO模型平移参数，不是重建腕关节坐标；图中使用stored units，物理尺度未独立验证。
- 三维诊断图使用缓存世界轴；官方视频显示前另做坐标轴旋转，两类图的屏幕方向不宜直接比较。
- 逐帧位移只在相邻两帧均有效、有限时计算，不跨越缺失段；它不是速度、抖动或误差指标。
- 框分数未作概率标定；模型渲染掩码面积不是分割IoU；最终有效不等于真实可见。
- 本次没有MPJPE、PVE、重投影误差或论文指标结果。后续需要匹配的标注、正确相机标定和正式评测协议，才能讨论精度。
- 目前覆盖3段4秒左右的作者示例和1段6秒外部输入，不能据此推断长视频、多人物、严重遮挡或其他相机条件的稳定性。

原始模型、MANO、推理缓存和服务器完整日志不随仓库分发。公开仓库存放精简视频、图片和数值记录；完整结果包另提供1080p导出，保留相同数据来源与限制。
