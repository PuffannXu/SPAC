# -*- coding: utf-8 -*-
"""ResNet*94.onnx 的 CPU / GPU 推理基准测试。

约束(对齐用户要求):
  - CPU、GPU 均限定单核/单设备
  - batch = 1 (一次处理一张图)
  - 延时与功耗均包含数据传输 (GPU 含 H2D + D2H)
  - 预热后正式测量, 多次重复取平均

测量:
  - 延时: 单次推理(含传输)的平均耗时 (ms)
  - 吞吐: FPS (帧/秒, 每秒处理的图像数) = 推理次数 / 墙钟时间
  - GPU 功耗: nvidia-smi 后台采样 power.draw (板级, 含核心+HBM), 取测量窗口均值
  - GPU 利用率: nvidia-smi utilization.gpu 采样均值
  - CPU 功耗: 加 --power, 经 LibreHardwareMonitor HTTP 接口实测 CPU Package Power
              (需 LHM 管理员运行 + Options->Remote Web Server)。读不到则自动降级。

用法:
  CPU: taskset 限 1 核 + 线程数=1
  GPU: CUDA_VISIBLE_DEVICES 指定单卡
"""
import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.request

import numpy as np

# LibreHardwareMonitor Remote Web Server (Options -> Remote Web Server -> Run)
LHM_URL = "http://localhost:8085/data.json"


def _walk_sensors(node, found):
    text = node.get("Text", "")
    value = node.get("Value", "")
    if value and isinstance(value, str) and value.strip().endswith("W"):
        found.append((text, value))
    for ch in node.get("Children", []):
        _walk_sensors(ch, found)


def read_cpu_power_lhm():
    """读 LibreHardwareMonitor HTTP 接口的 CPU Package Power (W)。读不到返回 None。"""
    try:
        with urllib.request.urlopen(LHM_URL, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8", "ignore"))
        found = []
        _walk_sensors(data, found)
        if not found:
            return None

        def to_w(s):
            m = re.search(r"[-+]?\d*\.?\d+", s.replace(",", "."))
            return float(m.group()) if m else None
        for name, val in found:
            if "package" in name.lower():
                w = to_w(val)
                if w is not None:
                    return w
        for name, val in found:
            ln = name.lower()
            if "cpu" in ln and ("total" in ln or "power" in ln):
                w = to_w(val)
                if w is not None:
                    return w
        ws = [to_w(v) for _, v in found]
        ws = [w for w in ws if w is not None]
        return max(ws) if ws else None
    except Exception:
        return None


def sample_cpu(stop_evt, out_list, interval=0.2):
    """后台线程: 周期采样 CPU Package Power (经 LHM)。"""
    while not stop_evt.is_set():
        p = read_cpu_power_lhm()
        if p is not None:
            out_list.append(p)
        time.sleep(interval)


def sample_gpu(stop_evt, gpu_index, out_list, interval=0.05):
    """后台线程: 周期采样指定 GPU 的功耗与利用率。"""
    import subprocess
    while not stop_evt.is_set():
        try:
            r = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=power.draw,utilization.gpu",
                 "--format=csv,noheader,nounits",
                 "-i", str(gpu_index)],
                capture_output=True, text=True, timeout=2)
            line = r.stdout.strip().splitlines()[0]
            p, u = [float(v) for v in line.split(",")]
            out_list.append((p, u))
        except Exception:
            pass
        time.sleep(interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--device", choices=["cpu", "gpu"], required=True)
    ap.add_argument("--gpu-index", type=int, default=0,
                    help="物理 GPU 编号(用于 nvidia-smi 采样, 与 CUDA_VISIBLE_DEVICES 对应)")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--power", action="store_true",
                    help="CPU: 经 LibreHardwareMonitor 实测功耗(需管理员运行+Remote Web Server); GPU: 默认即采样")
    args = ap.parse_args()

    import onnxruntime as ort
    print(f"onnxruntime {ort.__version__}")

    so = ort.SessionOptions()
    if args.device == "cpu":
        # 单核约束: 限制 ORT 线程
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
    else:
        # GPU: CUDA_VISIBLE_DEVICES 已限定单卡, device_id=0 指代该可见卡
        providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]

    sess = ort.InferenceSession(args.onnx, sess_options=so, providers=providers)
    print("active providers:", sess.get_providers())
    in_name = sess.get_inputs()[0].name
    in_shape = [d if isinstance(d, int) else 1 for d in sess.get_inputs()[0].shape]

    # batch=1 的随机输入 (新数据每次都重新交给 sess.run, 因此 GPU 路径天然含 H2D/D2H)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(in_shape).astype(np.float32)

    # 预热
    for _ in range(args.warmup):
        sess.run(None, {in_name: x})

    # 功耗采样线程
    samples = []
    stop_evt = threading.Event()
    sampler = None
    if args.device == "gpu":
        sampler = threading.Thread(target=sample_gpu,
                                   args=(stop_evt, args.gpu_index, samples))
        sampler.start()
        time.sleep(0.3)  # 让采样先跑起来
    elif args.device == "cpu" and args.power:
        p0 = read_cpu_power_lhm()
        if p0 is None:
            print("⚠️ 读不到 CPU 功耗: 需 LibreHardwareMonitor 管理员运行 + "
                  "Options->Remote Web Server (http://localhost:8085)。仍继续测延时/吞吐。")
        else:
            print(f"✓ CPU 功耗源就绪, 当前 Package Power = {p0:.1f} W")
            sampler = threading.Thread(target=sample_cpu, args=(stop_evt, samples))
            sampler.start()
            time.sleep(0.3)

    # 正式测量: 每次新建一份输入数组, 确保每次 run 都发生数据传输
    lat = np.empty(args.iters, dtype=np.float64)
    t0_all = time.perf_counter()
    for i in range(args.iters):
        xi = rng.standard_normal(in_shape).astype(np.float32)
        t0 = time.perf_counter()
        sess.run(None, {in_name: xi})  # 含 H2D + 计算 + D2H
        lat[i] = (time.perf_counter() - t0) * 1000.0  # ms
    wall = time.perf_counter() - t0_all

    if sampler is not None:
        stop_evt.set()
        sampler.join()

    mean_lat = float(lat.mean())
    p50 = float(np.percentile(lat, 50))
    p99 = float(np.percentile(lat, 99))
    fps = args.iters / wall  # 实测吞吐(墙钟), 帧/秒

    print("==== RESULT ====")
    print(f"device={args.device}")
    print(f"iters={args.iters} warmup={args.warmup} batch=1")
    print(f"mean_latency_ms={mean_lat:.4f}")
    print(f"p50_ms={p50:.4f} p99_ms={p99:.4f}")
    print(f"throughput_fps={fps:.2f}")
    if args.device == "gpu" and samples:
        ps = np.array([s[0] for s in samples])
        us = np.array([s[1] for s in samples])
        print(f"gpu_power_W_mean={ps.mean():.2f}")
        print(f"gpu_power_W_min={ps.min():.2f} gpu_power_W_max={ps.max():.2f}")
        print(f"gpu_util_pct_mean={us.mean():.2f}")
        print(f"gpu_power_samples={len(samples)}")
        print(f"energy_efficiency_FPS_per_W={fps/ps.mean():.4f}")
    elif args.device == "cpu" and args.power:
        if samples:
            ps = np.array(samples)
            print(f"cpu_power_W_mean={ps.mean():.2f}")
            print(f"cpu_power_W_min={ps.min():.2f} cpu_power_W_max={ps.max():.2f}")
            print(f"cpu_power_samples={len(samples)}")
            print(f"energy_efficiency_FPS_per_W={fps/ps.mean():.4f}")
        else:
            print("cpu_power=N/A (未采到样本, 检查 LibreHardwareMonitor)")


if __name__ == "__main__":
    main()
