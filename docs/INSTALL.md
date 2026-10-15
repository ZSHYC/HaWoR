# 安装与离屏渲染

以下命令均在仓库根目录执行。已验证平台为 Linux x86_64、Ubuntu 22.04、GCC 11.4、NVIDIA A800（sm_80）。需要可用的 NVIDIA 驱动、Bash、Git、Conda、C/C++ 编译器及系统 EGL/OpenGL 运行库；CUDA 工具链安装在独立 Conda 环境内。

## 1. 克隆完整源码

```bash
git clone --recursive https://github.com/ZSHYC/HaWoR.git
cd HaWoR
# 已经克隆过时，补齐嵌套子模块：
git submodule update --init --recursive
```

不要省略子模块：DROID-SLAM 编译依赖 Eigen 和 lietorch，包括 lietorch 内的嵌套依赖。复现时保留仓库记录的子模块提交。

## 2. 创建 Conda 环境

先在当前终端初始化自己的 Conda，例如 `source ~/miniforge3/etc/profile.d/conda.sh`，再执行：

```bash
CONDA_CHANNEL_PRIORITY=strict conda env create -f environment.yml
source scripts/server/activate.sh
python --version
nvcc --version
```

默认环境名为 `hawor`。如需按路径创建：

```bash
CONDA_CHANNEL_PRIORITY=strict conda env create --prefix /path/to/envs/hawor -f environment.yml
export HAWOR_ENV_PREFIX=/path/to/envs/hawor
export HAWOR_CONDA_ROOT=/path/to/miniforge3
source scripts/server/activate.sh
```

自定义设置也可写入仓库根目录的 `.env.local`；从 [.env.example](../.env.example) 复制后修改即可。该文件被 Git 忽略。脚本默认激活名为 `hawor` 的环境；设置 `HAWOR_ENV_PREFIX` 时按指定路径激活。

`environment.yml` 固定 CUDA 11.7 的编译器、运行库及开发库。使用 strict channel priority，避免旧 CUDA 元包混入更新的 CUDA 运行库。Linux 的完整 Conda 快照位于 `environment/conda-linux-64.lock.txt`，可替代上面的创建命令：

```bash
conda create --name hawor --file environment/conda-linux-64.lock.txt
source scripts/server/activate.sh
```

## 3. 安装 Python 依赖与 CUDA 扩展

必须先安装 PyTorch；PyTorch3D、torch-scatter 等源码构建依赖它。A800 的 CUDA 架构为 `8.0`，其他 GPU 应在安装前设置正确的 `TORCH_CUDA_ARCH_LIST`。DROID-SLAM 上游 setup 另有自己的架构列表。

```bash
python -m pip install numpy==1.26.4 torch==1.13.0+cu117 torchvision==0.14.0+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install --no-build-isolation -r environment/requirements.lock.txt
(cd thirdparty/DROID-SLAM && python setup.py install)
python -m pip check
```

`requirements.txt` 是便于维护的关键依赖列表；`environment/requirements.lock.txt` 记录验证环境的完整 Python 版本快照，推荐复现时使用后者。DROID-SLAM / lietorch 由固定版本的子模块单独构建，不包含在 pip 快照中。源码编译可能耗时十几分钟；不要在编译未结束时重复启动安装。

兼容性选择：

- PyTorch 1.13.0 + CUDA 11.7，NumPy 1.26.4。
- PyTorch3D v0.7.2、Chumpy 均固定到具体 Git 提交，避免浮动分支更新。
- `opencv-python` 与 `opencv-contrib-python-headless` 固定相同版本，以满足 Ultralytics / Supervision 与 AITViewer 的依赖声明。两者共享 `cv2` 命名空间，勿单独升级其中之一。
- Pillow 10.4.0、AITViewer 1.13.0、moderngl-window 2.4.6 与 ModernGL 5.10.0 使用本次实测组合。

## 4. 检查 EGL 离屏渲染

```bash
env -u DISPLAY python scripts/server/check_headless.py --output-dir /tmp/hawor-headless-check
```

成功时会输出 world / cam 两段视频的检查结果，并写入 `results.json`。该脚本使用合成网格，不需要权重、MANO 或官方示例推理；它会检查实际解码后的帧数、尺寸、帧率、网格颜色及运动。

激活脚本设置私有 `CUDA_HOME`、动态库路径、`PYOPENGL_PLATFORM=egl`、`MPLBACKEND=Agg` 与编译并行数。Linux 下 `--headless` 会为 AITViewer 显式选择 EGL 上下文；单独设置 `PYOPENGL_PLATFORM` 不足以完成这个选择。

`CUDA_VISIBLE_DEVICES` 用于 PyTorch / CUDA；EGL 默认设备选择不一定跟随该变量。多 GPU 机器可按本机 EGL 驱动配置选择渲染设备，不能据 CUDA 编号推断实际渲染 GPU。

## 常见问题

| 问题 | 检查方向 |
| --- | --- |
| 找不到 Conda 初始化脚本 | 在 `.env.local` 设置 `HAWOR_CONDA_ROOT`，或先初始化当前终端的 Conda |
| nvcc / CUDA 版本不一致 | 确认已 source 激活脚本，`CUDA_HOME` 与 `which nvcc` 均指向当前环境 |
| 找不到 Eigen 或 lietorch 源码 | 执行 `git submodule update --init --recursive` |
| EGL 上下文创建失败 | 检查 NVIDIA 驱动及系统 EGL/OpenGL 库；用 `check_headless.py` 单独定位 |
| SSH 上弹窗 / X11 错误 | 官方示例命令加 `--headless`，先通过无 DISPLAY 的渲染检查 |
| 修改输入后仍读到旧结果 | 同名目录会复用缓存；备份并移走旧目录，或更换输入文件名 |

环境就绪后按 [模型准备](MODELS.md) 安装模型，再运行 [README](../README.md) 中的示例命令。
