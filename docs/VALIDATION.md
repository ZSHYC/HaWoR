# 验证记录

验证日期：2026-10-15。范围为官方示例及外部短视频推理与 Linux EGL 离屏可视化，不涵盖完整训练或数据集评测。

## 环境

| 项目 | 实测版本 |
| --- | --- |
| 系统 / 编译器 | Ubuntu 22.04 x86_64 / GCC 11.4 |
| GPU | NVIDIA A800-SXM4-80GB，sm_80 |
| Python / PyTorch | 3.10.22 / 1.13.0+cu117 |
| CUDA 编译器 / 运行库 | 11.7.99 / 11.7.99，环境内独立安装 |
| TorchVision / NumPy | 0.14.0+cu117 / 1.26.4 |
| PyTorch3D / torch-scatter | 0.7.2 / 2.1.2 |
| AITViewer / ModernGL | 1.13.0 / 5.10.0 |

Conda 与 Python 快照在 [environment/](../environment/)。快照来自通过验证的已安装环境；尚未在第二台干净机器上完整重建。

## 已完成检查

- `conda activate hawor`：可按名字发现环境；Bash 初始化后在新登录终端可直接使用。
- `python scripts/server/check_conda.py`：真实 Conda 激活、重复激活、切换至 base、双向叠加环境均通过；退出后恢复原先未设置、空值、有值及未导出的变量。
- `python -m pip check`：依赖声明检查通过。
- 私有 CUDA 工具链：共享 CUDA 测试程序编译、链接与 GPU 执行通过，使用环境内 CUDA 11.7 库。
- torch-scatter CUDA、PyTorch3D CUDA 光栅化、DROID CUDA 相关性计算、lietorch SE3 CUDA exp/log 均通过实际运算。
- 六个公开模型 / 配置文件完成尺寸与 SHA-256 校验；HaWoR、infiller、DROID 权重可加载，Metric3D 完成实际单帧深度推理。
- MANO 左右手前向计算均得到 778 个顶点和 21 个关节。
- 无 DISPLAY 下，合成网格的 world / cam 导出均通过尺寸、帧数、帧率、颜色和运动检查，并验证重复导出覆盖。

## 多视频扩展记录

新增`segment_018`、`segment_037`和独立Open-AoE外部6秒输入，均完成完整推理与两种视角导出。四例的输入、原始导出和整理后视频已完整解码检查，预测五组数组均有限。定性观察、覆盖/补全统计、耗时、来源和可重跑命令见[多视频复现报告](RESULTS.md)。这些检查不构成准确率评价。

## 官方示例

输入为仓库内 `example/video_0.mp4`：1920×1080、121 帧、30 fps。已依次完成手部检测与跟踪、HaWoR、masked DROID-SLAM、Metric3D 尺度估计和运动补全，随后分别导出两种视角。

| 视角 | 输出 | 实际解码结果 |
| --- | --- | --- |
| world | `example/video_0/vis_world_0_120/aitviewer/video.mp4` | 1920×1080，120 帧，30 fps，4 秒 |
| cam | `example/video_0/vis_cam_0_120/aitviewer/video.mp4` | 1920×1080，120 帧，30 fps，4 秒 |

两种视频均完整解码检查，并抽帧查看了手部网格及相机视角下的视频背景。生成文件与含用户路径的完整运行日志不随仓库分发。

## 已知限制

- 上游 demo 的末端排他切片使本示例从 121 帧输入导出 120 帧，本 fork 保留该行为。
- world 视角保留上游默认观察相机，手部可能在画面中较小；cam 视角用于直接查看图像上的网格叠加。
- cam 模式仍执行上游 SLAM 流程，并非省略世界重建阶段的快捷模式。
- 默认 30 fps 导出；其他输入帧率、长视频、多段手部跟踪及其他 GPU 尚未系统验证。
- 离屏渲染已验证；本次服务器复现没有测试交互式桌面窗口。
- HOT3D / DexYCB 评测流程参考 [上游说明](UPSTREAM.md)，不能把示例运行结果视为论文指标复现。
