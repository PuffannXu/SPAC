# -*- coding: utf-8 -*-
"""CPU 全核并发基准: N 个进程各绑一个物理核, 各跑 batch=1 推理, 统计整机总吞吐 (FPS)。

公平对齐: 面积按整颗芯片(660 mm2, 64 物理核)计, 则吞吐也应让全部核心满载,
即整颗芯片 ↔ 整颗芯片满负载吞吐, 与 GPU 单卡整张跑同口径。

每个进程内部 batch=1 (一次一张图), 多进程并行 = 多张图同时处理, 保持单图语义。
"""
import argparse
import multiprocessing as mp
import os
import subprocess
import threading
import time

import numpy as np


import json
import re
import urllib.request

# LibreHardwareMonitor Remote Web Server 地址 (Options -> Remote Web Server -> Run)
LHM_URL = "http://localhost:8085/data.json"


def _walk_sensors(node, found):
    """递归遍历 LHM data.json 树, 收集 (名字, 数值字符串) 形如 ('CPU Package', '45.3 W')。"""
    text = node.get("Text", "")
    value = node.get("Value", "")
    if value and isinstance(value, str) and value.strip().endswith("W"):
        found.append((text, value))
    for ch in node.get("Children", []):
        _walk_sensors(ch, found)


def read_cpu_power_lhm():
    """读 LibreHardwareMonitor HTTP 接口的 CPU Package Power (W)。
    需 LHM 以管理员身份运行 + Options->Remote Web Server->Run。读不到返回 None。"""
    try:
        with urllib.request.urlopen(LHM_URL, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8", "ignore"))
        found = []
        _walk_sensors(data, found)
        if not found:
            return None
        # 优先匹配 CPU Package; 退而求其次匹配含 Package 的; 再不行取最大功耗项
        def to_w(s):
            m = re.search(r"[-+]?\d*\.?\d+", s.replace(",", "."))
            return float(m.group()) if m else None
        # 1) 名字含 'package' (不区分大小写)
        for name, val in found:
            if "package" in name.lower():
                w = to_w(val)
                if w is not None:
                    return w
        # 2) 名字含 'cpu' 且含 'power'/'total'
        for name, val in found:
            ln = name.lower()
            if "cpu" in ln and ("total" in ln or "power" in ln):
                w = to_w(val)
                if w is not None:
                    return w
        # 3) 兜底: 所有 W 传感器里取最大值 (通常就是 CPU 整体)
        ws = [to_w(v) for _, v in found]
        ws = [w for w in ws if w is not None]
        return max(ws) if ws else None
    except Exception:
        return None


def power_sampler(stop_evt, out_list, interval=0.2):
    """后台线程: 周期采样 CPU Package Power。"""
    while not stop_evt.is_set():
        p = read_cpu_power_lhm()
        if p is not None:
            out_list.append(p)
        time.sleep(interval)


def worker(core_id, onnx_path, warmup, duration, ret_q):
    # 绑定到指定物理核
    try:
        os.sched_setaffinity(0, {core_id})
    except Exception:
        pass
    os.environ["OMP_NUM_THREADS"] = "1"
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(onnx_path, sess_options=so,
                                providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    in_shape = [d if isinstance(d, int) else 1 for d in sess.get_inputs()[0].shape]
    rng = np.random.default_rng(core_id)
    x = rng.standard_normal(in_shape).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {in_name: x})
    # 固定时长内尽可能多地推理
    cnt = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration:
        xi = rng.standard_normal(in_shape).astype(np.float32)
        sess.run(None, {in_name: xi})
        cnt += 1
    wall = time.perf_counter() - t0
    ret_q.put((core_id, cnt, wall))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--cores", type=int, default=64, help="并发进程数(物理核数)")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--duration", type=float, default=20.0, help="每进程测量时长(秒)")
    ap.add_argument("--power", action="store_true",
                    help="同时采样 CPU 功耗(需 LibreHardwareMonitor 以管理员运行中)")
    args = ap.parse_args()

    # 启动前先探测功耗源
    if args.power:
        p0 = read_cpu_power_lhm()
        if p0 is None:
            print("⚠️ 读不到 CPU 功耗: 请确认 LibreHardwareMonitor 已【管理员身份运行】"
                  "且已开启 Options->Remote Web Server (http://localhost:8085)。仍继续测吞吐。")
        else:
            print(f"✓ 功耗源就绪, 当前 CPU Package Power = {p0:.1f} W")

    print(f"启动 {args.cores} 个进程, 每进程绑1物理核, batch=1, 各测 {args.duration}s")

    # 功耗采样线程
    samples = []
    stop_evt = threading.Event()
    sampler = None
    if args.power:
        sampler = threading.Thread(target=power_sampler, args=(stop_evt, samples))
        sampler.start()

    ret_q = mp.Queue()
    procs = []
    # 物理核 0..N-1
    for c in range(args.cores):
        p = mp.Process(target=worker,
                       args=(c, args.onnx, args.warmup, args.duration, ret_q))
        p.start()
        procs.append(p)
    results = [ret_q.get() for _ in procs]
    for p in procs:
        p.join()

    if sampler is not None:
        stop_evt.set()
        sampler.join()

    total_cnt = sum(r[1] for r in results)
    max_wall = max(r[2] for r in results)
    total_fps = total_cnt / max_wall
    per_core_fps = [r[1] / r[2] for r in results]

    print("==== RESULT (CPU all-cores) ====")
    print(f"cores={args.cores} duration={args.duration}s")
    print(f"total_inferences={total_cnt}")
    print(f"aggregate_throughput_fps={total_fps:.2f}")
    print(f"per_core_fps_mean={np.mean(per_core_fps):.2f} "
          f"min={np.min(per_core_fps):.2f} max={np.max(per_core_fps):.2f}")
    if args.power and samples:
        parr = np.array(samples)
        pmean = parr.mean()
        print(f"cpu_power_W_mean={pmean:.2f}")
        print(f"cpu_power_W_min={parr.min():.2f} cpu_power_W_max={parr.max():.2f}")
        print(f"cpu_power_samples={len(samples)}")
        print(f"energy_efficiency_FPS_per_W={total_fps/pmean:.4f}")
    elif args.power:
        print("cpu_power=N/A (未采到样本, 检查 LibreHardwareMonitor)")


if __name__ == "__main__":
    main()
