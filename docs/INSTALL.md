# 安装与离屏渲染

以下命令均在仓库根目录执行。已验证平台为 Linux x86_64、Ubuntu 22.04、GCC 11.4、NVIDIA A800（sm_80）。需要可用的 NVIDIA 驱动、Bash、Git、Conda、C/C++ 编译器及系统 EGL/OpenGL 运行库；CUDA 工具链安装在独立 Conda 环境内。

创建与激活环境采用普通 Conda 命令。相较上游，环境文件固定了已验证的版本，另提供 Conda 原生激活 / 退出钩子，自动设置和恢复 CUDA 与 EGL 变量。

## 1. 克隆完整源码

```bash
git clone --recursive https://github.com/ZSHYC/HaWoR.git
cd HaWoR
# 已经克隆过时，补齐嵌套子模块：
git submodule update --init --recursive
```

不要省略子模块：DROID-SLAM 编译依赖 Eigen 和 lietorch，包括 lietorch 内的嵌套依赖。复现时保留仓库记录的子模块提交。

## 2. 创建 Conda 环境

如果当前终端尚未初始化 Conda，先按自己的安装位置执行，例如：

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda init bash
```

`conda init bash` 只需配置一次，之后重新打开终端即可使用 Conda。环境创建与激活命令为：

```bash
CONDA_CHANNEL_PRIORITY=strict conda env create -f environment.yml
conda activate hawor
```

默认环境名为 `hawor`，位置由 Conda 的 `envs_dirs` 决定。Conda 安装目录与项目代码目录无需一致。`environment.yml` 固定 CUDA 11.7 的编译器、运行库及开发库；使用 strict channel priority，避免旧 CUDA 元包混入更新的 CUDA 运行库。

Linux 的完整 Conda 快照位于 `environment/conda-linux-64.lock.txt`，可用以下命令替代上面的创建命令：

```bash
conda create --name hawor --file environment/conda-linux-64.lock.txt
conda activate hawor
```

如需指定存放位置，可改用 `--prefix /path/to/envs/hawor` 创建，并用 `conda activate /path/to/envs/hawor` 激活。如果希望之后仍能按名字激活，将其父目录加入搜索路径：

```bash
conda config --prepend envs_dirs /path/to/envs
conda activate hawor
```

环境已存在时无需再次创建。旧版项目的 `source scripts/server/activate.sh` 已改为这里的普通 Conda 激活方式。

## 3. 安装 Conda 激活钩子（每个环境一次）

在目标环境已激活时，从仓库根目录执行：

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
cp environment/conda/activate.d/hawor.sh "$CONDA_PREFIX/etc/conda/activate.d/hawor.sh"
cp environment/conda/deactivate.d/hawor.sh "$CONDA_PREFIX/etc/conda/deactivate.d/hawor.sh"
conda deactivate
conda activate hawor
python --version
nvcc --version
```

按完整路径激活的环境，最后一条激活命令也使用其完整路径。更新仓库中的钩子后，需要再次复制到环境中；退出旧激活状态后重新激活即可生效。

之后日常运行只需要：

```bash
conda activate hawor
# 运行推理或检查命令
conda deactivate
```

钩子由 Conda 自动调用，不要手动 source。它设置当前环境的 `CUDA_HOME`、动态库路径、`PYOPENGL_PLATFORM=egl`、`MPLBACKEND=Agg`。`TORCH_CUDA_ARCH_LIST` 默认 `8.0`，`MAX_JOBS` 与 `OMP_NUM_THREADS` 默认 `8`，已有非空值会保留。退出或切换环境后，原值、空值、未设置状态及导出属性会恢复。

## 4. 安装 Python 依赖与 CUDA 扩展

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

## 5. 验证激活与 EGL 离屏渲染

```bash
python scripts/server/check_conda.py
python -m pip check
env -u DISPLAY python scripts/server/check_headless.py --output-dir /tmp/hawor-headless-check
```

`check_conda.py` 在独立 Bash 子进程中实际调用 Conda，检查激活、重复激活、切换、叠加环境及变量恢复，不改变当前终端的激活状态。可传 `--conda-root /path/to/miniforge3` 或 `--env /path/to/envs/hawor`。

渲染检查成功时会输出 world / cam 两段视频的检查结果，并写入 `results.json`。该脚本使用合成网格，不需要权重、MANO 或官方示例推理；它会检查实际解码后的帧数、尺寸、帧率、网格颜色及运动。

Linux 下 `--headless` 会为 AITViewer 显式选择 EGL 上下文；单独设置 `PYOPENGL_PLATFORM` 不足以完成这个选择。`CUDA_VISIBLE_DEVICES` 用于 PyTorch / CUDA；EGL 默认设备选择不一定跟随该变量，不能据 CUDA 编号推断实际渲染 GPU。

## 常见问题

| 问题 | 检查方向 |
| --- | --- |
| 登录后找不到 conda | 执行 Conda 初始化，重新打开终端 |
| `conda activate hawor` 找不到环境 | 用 `conda info --envs` 检查位置，将环境父目录加入 `envs_dirs` 或按完整路径激活 |
| nvcc / CUDA 版本不一致 | 确认已安装上述钩子并重新激活，`CUDA_HOME` 与 `which nvcc` 均应指向当前环境 |
| 找不到 Eigen 或 lietorch 源码 | 执行 `git submodule update --init --recursive` |
| EGL 上下文创建失败 | 检查 NVIDIA 驱动及系统 EGL/OpenGL 库；用 `check_headless.py` 单独定位 |
| SSH 上弹窗 / X11 错误 | 官方示例命令加 `--headless`，先通过无 DISPLAY 的渲染检查 |
| 修改输入后仍读到旧结果 | 同名目录会复用缓存；备份并移走旧目录，或更换输入文件名 |

环境就绪后按 [模型准备](MODELS.md) 安装模型，再运行 [README](../README.md) 中的示例命令。
