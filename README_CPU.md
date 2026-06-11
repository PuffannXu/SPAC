# CPU 推理评测 —— 台式机使用说明

本目录包含在你自己的台式机上评测 ResNet18 ONNX **CPU 推理性能**所需的全部文件。

## 目录文件

| 文件 | 用途 |
|---|---|
| `ResNet18_128_128_128_int8_94.onnx` | 被测模型（输入 `[1,4,32,32]` float32，输出 `[1,10]`） |
| `bench_onnx.py` | **单核** batch=1 基准（延时 / 吞吐） |
| `bench_cpu_allcores.py` | **全核并发** 基准（整机吞吐） |
| `benchmark_cpu_gpu.md` | 服务器上的完整评测报告（含 GPU 对比，供参考） |

## 1. 环境准备

只需 Python 3 + 两个包：

```bash
pip install onnxruntime numpy
```

> 注意：装 `onnxruntime`（CPU 版）即可，**不要**装 `onnxruntime-gpu`，否则可能默认走 GPU。

## 2. 单核评测（延时 + 吞吐）

把进程绑到 **1 个 CPU 核**，batch=1 逐张推理：

### Linux / macOS
```bash
# Linux 用 taskset 绑核
taskset -c 0 python bench_onnx.py \
    --onnx ResNet18_128_128_128_int8_94.onnx --device cpu \
    --warmup 300 --iters 5000

# macOS 无 taskset，直接跑（脚本内已限 ORT 单线程）
python bench_onnx.py --onnx ResNet18_128_128_128_int8_94.onnx --device cpu \
    --warmup 300 --iters 5000
```

### Windows（PowerShell / CMD）
```powershell
python bench_onnx.py --onnx ResNet18_128_128_128_int8_94.onnx --device cpu --warmup 300 --iters 5000
```
> Windows 无 `taskset`，脚本已通过 `intra_op_num_threads=1` / `inter_op_num_threads=1` 限定单线程，
> 等效单核负载。若想强制绑核，可用 `start /affinity 1 python ...`。

**输出示例：**
```
mean_latency_ms=2.9151      # 单图平均延时
p50_ms=2.83 p99_ms=4.22     # 延时分位
throughput_fps=335.06       # 单核吞吐（帧/秒）
```

## 3. 全核评测（整机吞吐）

启动 N 个进程（N = 你的物理核数），每进程绑一个核、各跑 batch=1，统计整机聚合吞吐：

```bash
# 把 --cores 改成你台式机的物理核数（不是线程数）。例如 8 核 i7：
python bench_cpu_allcores.py \
    --onnx ResNet18_128_128_128_int8_94.onnx \
    --cores 8 --warmup 100 --duration 20
```

> - `--cores`：并发进程数，建议设为**物理核数**（避免超线程虚高）。查物理核：
>   - Linux: `lscpu | grep "Core(s) per socket"` × socket 数
>   - Windows: 任务管理器 → 性能 → CPU → "内核"
>   - macOS: `sysctl -n hw.physicalcpu`
> - `--duration`：每进程测量秒数。
> - Windows 上 `os.sched_setaffinity` 不可用，脚本会自动跳过绑核（吞吐仍可测，略有调度抖动）。

**输出示例：**
```
aggregate_throughput_fps=2528.02   # 全核聚合吞吐
per_core_fps_mean=316.08           # 每核平均吞吐
```

## 4. 功耗与能效（可选）

脚本本身**不测 CPU 功耗**（需要 root + RAPL/PCM）。若你想要能效（FPS/W）：

- **查你 CPU 的 TDP**（Intel ARK / CPU 包装），用 `能效 = 吞吐(FPS) / TDP(W)` 估算；
- 或在台式机上用 **HWiNFO64 / Intel Power Gadget**（Windows）实时读 CPU Package Power，
  推理时记录平均功耗，再算 `FPS / W`。这比 TDP 估算准。

## 参考：服务器实测数据（对照用）

服务器 CPU = Intel Xeon Platinum 8338C（Ice Lake-SP, 64 物理核）：

| | 数值 |
|---|---|
| 单核延时 | 2.92 ms |
| 单核吞吐 | 335 FPS |
| 全 64 核吞吐 | 15129 FPS |

你台式机核数更少、单核主频可能更高，**单核延时可能更低，但全核吞吐会低于服务器**，属正常。
