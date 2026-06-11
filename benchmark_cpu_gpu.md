# ResNet18 (ResNet*94.onnx) CPU / GPU 推理基准评测

测试模型：`ResNet18_128_128_128_int8_94.onnx`
输入：`[1, 4, 32, 32]` (float32)，输出：`[1, 10]`
推理框架：ONNX Runtime 1.24.3
评测日期：2026-06-08
评测脚本：`bench_onnx.py`（代码与本报告同目录）

> 说明：本评测仅衡量**推理性能**（延时/吞吐/功耗/能效），与模型精度无关。
> 评测约束（与对比对象对齐）：**CPU、GPU 均限定单核/单设备**，**batch = 1**（一次处理一张图），
> **延时与功耗均包含数据传输**（GPU 含 Host→Device 与 Device→Host 拷贝）。

---

## 结果汇总

| 指标 | CPU（单核） | GPU（单卡，含数据传输） |
|---|---|---|
| 设备型号 | Intel® Xeon® Platinum 8338C @ 2.60 GHz | NVIDIA A100-SXM4-80GB |
| 使用核心 / 设备 | 1 物理核（共 64 核 / 128 线程） | 1 张 A100（共 8 张） |
| 平均利用率 | 单核满载（1/64 ≈ 1.6% 整机） | 69.1%（单卡） |
| 平均延时 (mean) | **2.9151 ms** | **0.4147 ms** |
| 延时 p50 / p99 | 2.83 / 4.22 ms | 0.414 / 0.424 ms |
| 吞吐量 | **335.06 FPS** | **2100.82 FPS** |
| 平均功耗 | ≈ 6.4 W（TDP 估算，见下） | **101.71 W**（nvidia-smi 实测，板级） |
| 能效 | ≈ 52.4 FPS/W（基于 TDP 估算） | **20.65 FPS/W**（实测） |

> 吞吐量以 **FPS**（Frames Per Second，帧/秒，即每秒处理的图像数）衡量，为图像推理任务的标准指标；能效对应 **FPS/W**。

**对比：** 在单核 / 单卡、batch=1、含数据传输的条件下，单张 A100 的单图推理延时约为单核 CPU 的 **1/7**（0.41 ms vs 2.92 ms），吞吐约为 **6.3 倍**（2101 vs 335 FPS）。

---

## 功耗（单独列出）

| 设备 | 平均功耗 | 测量方式 |
|---|---|---|
| CPU（单核） | **≈ 6.4 W** | TDP 单核分摊估算（205 W ÷ 32 核），非实测 |
| GPU（单卡 A100） | **101.71 W** | nvidia-smi 实测板级功耗（含 GPU 核心 + HBM），idle 基线 ≈ 69 W |

## 算力能效 TOPS/W（单独列出）

模型单张图（`[1,4,32,32]`）推理的计算量为 **206.19 M MACs = 0.4124 GFLOPs**（按 1 MAC = 2 OPs；由 ONNX 图中全部 Conv/MatMul 节点逐层累加得到）。

**有效算力 = 单图 OPs × 吞吐(FPS)**，**算力能效 = 有效算力 ÷ 功耗**：

| 设备 | 单图算力 | 有效算力 (实测) | 算力能效 TOPS/W | 算力能效 GOPS/W |
|---|---|---|---|---|
| CPU（单核） | 0.4124 GFLOPs | 0.138 TOPS | **≈ 0.0216 TOPS/W**（估算）| 21.59 GOPS/W |
| GPU（单卡 A100） | 0.4124 GFLOPs | 0.866 TOPS | **≈ 0.0085 TOPS/W**（实测）| 8.52 GOPS/W |

> **关键说明：** 此处 TOPS/W 是 **batch = 1 单图场景下的「有效算力能效」**，即实际跑出来的算力 ÷ 功耗，**远低于硬件理论峰值**（A100 FP16 峰值约 312 TFLOPS）。原因是单张 32×32 小图的算力（0.4 GFLOPs）极小，完全填不满 GPU 的并行算力，板级功耗中又含约 69 W 的设备空闲基线——因此 GPU 的 TOPS/W 反而低于单核 CPU。这正是"一次处理一张图"对比场景的真实特性：大算力硬件在小 batch、小模型下能效受限。若需逼近峰值 TOPS/W，应增大 batch 或模型规模。CPU 侧的 TOPS/W 因功耗为 TDP 估算，同样仅供数量级参考。

---

## CPU 延时与吞吐评测

我们在一颗 **Intel® Xeon® Platinum 8338C CPU @ 2.60 GHz**（32 核/颗 × 2 颗，共 64 物理核 / 128 线程，1 TiB RAM）上评测了 CPU 推理性能。推理程序基于 Python + ONNX Runtime 1.24.3，使用 `CPUExecutionProvider`。为对齐"一次处理一张图"的对比对象，**强制限定单核**：通过 `taskset -c 0` 将进程绑定到单个物理核，并设置 ONNX Runtime `intra_op_num_threads = 1`、`inter_op_num_threads = 1`。所有推理均以 **batch = 1** 执行，延时包含输入数据从 NumPy 数组送入推理会话的完整过程。延时使用 Python 内置 `time.perf_counter` 测量。正式测试前先以 300 次推理预热以消除冷启动波动，随后进行 5000 次推理取均值。每次推理均重新生成一份随机输入图像（`[1,4,32,32]`），确保测得的延时反映真实单图处理开销。实测每次推理平均延时 **2.9151 ms**（p50 = 2.83 ms，p99 = 4.22 ms），对应吞吐 **335.06 FPS**（帧/秒）。

## CPU 能效评测

由于本评测平台**不具备 root 权限**，无法读取 Intel RAPL 能耗计数器（`/sys/class/powercap/intel-rapl/.../energy_uj` 为 root 只读），亦无 Intel® Performance Counter Monitor (PCM)、turbostat、perf energy 等工具，**CPU 功耗无法直接实测**。因此 CPU 功耗采用 **TDP 单核分摊估算**：Xeon Platinum 8338C 整颗 TDP 为 205 W，32 核分摊后单核约 **6.4 W**。据此估算能效约为 **52.4 FPS/W**。

> ⚠️ CPU 功耗与能效为**估算值**，非实测；仅供数量级参考。实测需在具备 root 权限的环境下使用 RAPL / PCM 采集板级功耗。

## GPU 延时与吞吐评测

我们在一张 **NVIDIA A100-SXM4-80GB**（80 GB HBM2e，400 W 功耗上限，驱动 575.57.08）上评测了 GPU 推理性能。推理程序基于 Python + ONNX Runtime 1.24.3，使用 `CUDAExecutionProvider`，通过 `CUDA_VISIBLE_DEVICES` 限定为**单张空闲 GPU**。所有推理均以 **batch = 1** 执行。为保证延时**包含数据传输**，每次推理都重新生成一份输入数组交给 `session.run()`，因此每次调用都包含 Host→Device 输入拷贝、核函数计算、以及 Device→Host 输出回传的完整链路。延时使用 Python 内置 `time.perf_counter` 测量。正式测试前以 500 次推理预热以消除冷启动与时钟爬升的影响，随后进行 5000 次推理取均值。实测每次推理平均延时 **0.4147 ms**（p50 = 0.414 ms，p99 = 0.424 ms），对应吞吐 **2100.82 FPS**（帧/秒）。

## GPU 能效评测

GPU 能耗使用 **NVIDIA System Management Interface (`nvidia-smi`)** 监测，该接口测量的是**板级总功耗**（包含 GPU 核心与 HBM 显存）。在推理过程中由后台线程以 ~50 ms 间隔周期性采样 `power.draw` 与 `utilization.gpu`，对测量窗口内的采样取均值。推理过程中平均功耗为 **101.71 W**，平均 GPU 利用率 **69.1%**，由此得到能效 **20.65 FPS/W**。

> 说明：在 batch = 1 的单图推理场景下，A100 的算力远未被填满，板级功耗中包含较大比例的设备空闲基线（idle ≈ 69 W）。这正是"一次处理一张图"场景的真实特性——大算力 GPU 在小 batch 下能效相对受限。

---

## 复现方式

```bash
# 环境: /data/programs/conda_envs/sglang (onnxruntime-gpu 1.24.3)
PY=/data/programs/conda_envs/sglang/bin/python

# CPU (单核绑定 + 单线程)
taskset -c 0 $PY bench_onnx.py \
    --onnx ResNet18_128_128_128_int8_94.onnx --device cpu \
    --warmup 300 --iters 5000

# GPU (单卡, 含 H2D/D2H 传输)
CUDA_VISIBLE_DEVICES=6 $PY bench_onnx.py \
    --onnx ResNet18_128_128_128_int8_94.onnx --device gpu --gpu-index 6 \
    --warmup 500 --iters 5000
```

## 测量方法学要点

- **单核/单设备**：CPU 用 `taskset` + ORT 单线程；GPU 用 `CUDA_VISIBLE_DEVICES` 锁单卡。
- **batch = 1**：逐张推理，匹配"一次处理一张图"的对比对象；可统计多张取平均。
- **含数据传输**：每次 `run()` 均传入新数组，GPU 路径天然包含 H2D + D2H。
- **预热**：CPU 300 次 / GPU 500 次，消除冷启动与 GPU 时钟爬升。
- **重复次数**：各 5000 次取均值，并报告 p50 / p99 反映稳定性。
- **功耗**：GPU 用 `nvidia-smi` 实测板级功耗（含核心+HBM）；CPU 因无 root 用 TDP 估算（已标注）。

---

# 面积归一化性能对比（Area-Normalised Comparison）

> 由于加速器吞吐直接受**硅面积**与**工艺节点**影响，参照论文范式给出两级归一化：
> 面积归一化（FPS/mm²、FPS/W/mm²）与「面积 + 工艺」归一化（FPS/mm²·nm²）。

## 芯片规格（die area / technology）

| | CPU | GPU |
|---|---|---|
| 型号 | Intel® Xeon® Platinum 8338C（Ice Lake-SP, XCC） | NVIDIA A100-SXM4-80GB（GA100） |
| 工艺 (nm) | **10**（Intel 10nm SuperFin / 10ESF） | **7**（TSMC N7） |
| Die 面积 (mm²) | **660**（整颗 XCC die，40 核满配；8338C 屏蔽至 32 核/颗）| **826**（NVIDIA 官方白皮书确认） |

> 面积来源：A100 = 826 mm²（NVIDIA Ampere 架构白皮书、Wikipedia 一致）；
> Ice Lake-SP XCC ≈ 660 mm²（WikiChip，业界通用引用值）。
> **口径说明**：CPU 与 GPU 均按**整颗芯片面积**计，与论文中 GPU 取整颗 826 mm² 的做法一致。

## 性能对比表

> ⚠️ **负载口径不对等，务必注意**：为与「整颗芯片面积」匹配，CPU 采用**全 64 物理核满载**
> （每核独立 batch=1 推理，多核并行 = 多图同时处理，保持单图语义）；GPU 按用户要求**保持单卡 batch=1 单图**
> （利用率仅 ~69%，算力未填满）。因此下表 CPU 是「整芯片满负载」、GPU 是「单图」，两者负载强度不同。

| 指标 | CPU（全 64 核满载） | GPU（单卡, batch=1） |
|---|---|---|
| 工艺 (nm) | 10 | 7 |
| Die 面积 (mm²) | 660 | 826 |
| 功耗 (W) | **205**（整颗 TDP，满载估算） | **101.71**（nvidia-smi 实测板级） |
| 延时 (ms) | 单核 ~2.92（满核单图延时见正文） | 0.4147 |
| 吞吐 (FPS) | **15128.84** | **2100.82** |
| 能效 (FPS/W) | **73.80** | **20.66** |
| Norm. 吞吐 (FPS/mm²) | **22.92** | **2.54** |
| Norm. 吞吐 (FPS/mm²·nm²) | **0.2292** | **0.0519** |
| Norm. 能效 (FPS/W/mm²) | **0.1118** | **0.0250** |

## 倍数对比（CPU 相对 GPU）

| 指标 | CPU / GPU |
|---|---|
| 吞吐 | **7.20×** |
| 能效 (FPS/W) | **3.57×** |
| 面积归一化吞吐 (FPS/mm²) | **9.01×** |
| 面积+工艺归一化吞吐 (FPS/mm²·nm²) | **4.42×** |
| 面积归一化能效 (FPS/W/mm²) | **4.47×** |

## 结论与口径说明

在本场景（ResNet18 小模型、32×32 小图、INT8 结构）下，**全核 CPU 在吞吐、能效及面积归一化指标上全面优于单卡 A100**。
根本原因：单张小图的算力（0.41 GFLOPs）远不足以填满 A100 的并行算力，且 GPU 板级功耗含约 69 W 空闲基线；
而 64 核 CPU 以多核并行处理多图，硬件利用率更充分。这印证了「大算力 GPU 在小 batch / 小模型场景下能效受限」。

**结果解读需注意以下口径：**
1. **负载不对等**：CPU 全核满载 vs GPU 单图 batch=1。若 GPU 也用大 batch 跑满，其吞吐与能效会显著提升，对比结论可能反转。本表反映的是「整芯片满载 CPU」对「单图 GPU」的特定场景。
2. **CPU 功耗为 TDP 估算**（满载 205 W），非实测（本机无 root，RAPL/PCM 不可用）。
3. **CPU 单核延时 2.92 ms** 仍是单图延时的代表值；多核满载提升的是吞吐而非单图延时。

## CPU 全核基准复现

```bash
PY=/data/programs/conda_envs/sglang/bin/python
$PY bench_cpu_allcores.py \
    --onnx ResNet18_128_128_128_int8_94.onnx \
    --cores 64 --warmup 100 --duration 20
```
