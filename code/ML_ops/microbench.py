#!/usr/bin/env python3
"""microbench.py -- one kernel group, looping back-to-back, sustained.

    python3 microbench.py --op bn_bwd --shape s1 --dtype bf16 \
        --seconds 60 --discard 15 --axis clk --axisval 1140 --outdir <dir> \
        [--nvtx]

bn_bwd and conv_bwd build their forward graph once outside the timed loop,
and call autograd.grad(..., retain_graph=True) in the loop. gemm, layernorm, sdpa
and pool are fwd+bwd.  Instrumentation (NVML sampler, trapezoid
energy, PCI-bus device resolution) is in this file, the benchmark is
a single self-contained script.

Outputs, all in --outdir:
    timeline_gpu<N>.csv   100 Hz NVML samples: kind,t_ns,value
    iters_gpu<N>.csv      iter,t_start_ns,t_end_ns,retained
    summary.json          the run record
"""
import argparse
import json
import os
import statistics
import sys
import pynvml
import time

import threading
MONO = time.monotonic_ns


class Sampler(threading.Thread):
    """100 Hz NVML sampler for ONE physical GPU: kind,t_ns,value rows."""

    def __init__(self, handle, path, hz=100.0):
        super().__init__(daemon=True)
        self.h, self.path, self.period = handle, path, 1.0 / hz
        self.stop = threading.Event()
        self.n = 0

    def run(self):
        f = open(self.path, "w", buffering=1 << 20)
        f.write("kind,t_ns,value\n")
        g = pynvml
        nxt = time.monotonic()
        try:
            while not self.stop.is_set():
                t = MONO()
                try:
                    p = g.nvmlDeviceGetPowerUsage(self.h) / 1000.0
                    sm = g.nvmlDeviceGetClockInfo(self.h, g.NVML_CLOCK_SM)
                    mc = g.nvmlDeviceGetClockInfo(self.h, g.NVML_CLOCK_MEM)
                    rs = g.nvmlDeviceGetCurrentClocksThrottleReasons(self.h)
                    tc = g.nvmlDeviceGetTemperature(self.h, g.NVML_TEMPERATURE_GPU)
                except g.NVMLError:
                    nxt += self.period
                    self.stop.wait(max(0.0, nxt - time.monotonic()))
                    continue
                f.write(f"power,{t},{p}\nclock,{t},{sm}\nmemclock,{t},{mc}\n"
                        f"reasons,{t},{rs}\ntemp,{t},{tc}\n")
                self.n += 1
                nxt += self.period
                self.stop.wait(max(0.0, nxt - time.monotonic()))
        finally:
            f.close()


def visible_physical_indices():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        return [int(t) for t in cvd.split(",") if t.strip() != ""]
    return list(range(pynvml.nvmlDeviceGetCount()))


def bus_id_of(cuda_dev):
    """PCI bus id of the cuda-visible device, from NVML (torch's device
    properties lost pci_bus_id/pci_domain_id on some builds, so we do not
    rely on torch for this at all)."""
    phys = visible_physical_indices()[cuda_dev]
    return pynvml.nvmlDeviceGetPciInfo(
        pynvml.nvmlDeviceGetHandleByIndex(phys)).busId


def handle_for(phys_idx, bus=None):
    """Prefer the PCI bus id -- an ordering assumption is the one thing that
    has silently mislabelled a device here before."""
    if bus:
        try:
            h = pynvml.nvmlDeviceGetHandleByPciBusId(bus.encode())
            return h, pynvml.nvmlDeviceGetIndex(h), "pci"
        except Exception:
            pass
    return pynvml.nvmlDeviceGetHandleByIndex(phys_idx), phys_idx, "index"


def trapezoid_energy(samples, t0, t1, max_gap_ns=int(0.5e9)):
    """Trapezoid-integrate (t_ns, watts) over [t0,t1], skipping gaps > 0.5 s.

    Endpoints are interpolated onto t0 and t1 so the window is exactly the
    measured-step window; a plain sum over the samples that happen to land
    inside it loses up to one sample period at each end.
    """
    pts = sorted(t for t in samples if t0 - max_gap_ns <= t[0] <= t1 + max_gap_ns)
    if len(pts) < 2:
        return float("nan"), 0
    e, skipped = 0.0, 0
    for (ta, va), (tb, vb) in zip(pts, pts[1:]):
        if tb <= t0 or ta >= t1:
            continue
        if tb - ta > max_gap_ns:
            skipped += 1
            continue
        sa, sb, wa, wb = ta, tb, va, vb
        if sa < t0:
            wa = va + (vb - va) * (t0 - ta) / (tb - ta)
            sa = t0
        if sb > t1:
            wb = va + (vb - va) * (t1 - ta) / (tb - ta)
            sb = t1
        e += 0.5 * (wa + wb) * (sb - sa) * 1e-9
    return e, skipped


def read_timeline(path, t0, t1):
    power, clocks, reasons = [], [], []
    with open(path) as f:
        next(f)
        for line in f:
            k, t, v = line.split(",")
            t = int(t)
            if k == "power":
                power.append((t, float(v)))
            elif k == "clock" and t0 <= t <= t1:
                clocks.append(int(v))
            elif k == "reasons" and t0 <= t <= t1:
                reasons.append(int(v))
    return power, clocks, reasons


# ResNet-50 stage activation shapes, (C,H,W).
SHAPES = {
    "s1": (64, 112, 112),     # stem / layer1
    "s2": (256, 56, 56),      # layer2 in
    "s3": (1024, 14, 14),     # layer4 in
}
VIT = {"seq": 197, "dim": 768, "heads": 12, "mlp": 3072}

DTYPES = ("bf16", "fp32", "tf32", "fp64")

CONV_OPS = ("conv_fwd", "conv_bwd")
VIT_OPS = ("gemm", "layernorm", "sdpa")
OPS = ("bn_fwd", "bn_bwd", "conv_fwd", "conv_bwd", "elementwise",
       "pool") + VIT_OPS


def build_op(name, shape, dtype, N, dev):
    """(op, meta).  op() runs the op ONCE; everything it touches is
    allocated here, so the timed loop never allocates."""
    import torch
    import torch.nn.functional as F

    # tf32/fp32 are the same tensors; only the global flags differ (main sets
    # them before anything is built).  Params/running stats stay fp32 under
    # bf16/tf32, what autocast does in a real model
    td = {"bf16": torch.bfloat16, "fp64": torch.float64}.get(
        dtype, torch.float32)
    pd = torch.float64 if dtype == "fp64" else torch.float32
    meta = {"op": name, "dtype": dtype, "N": N,
            "param_dtype": str(pd).replace("torch.", "")}

    if name in VIT_OPS:
        S, D, H, M = VIT["seq"], VIT["dim"], VIT["heads"], VIT["mlp"]
        meta.update(VIT)
        if name == "gemm":
            x = torch.randn(N, S, D, device=dev, dtype=td, requires_grad=True)
            w = torch.randn(M, D, device=dev, dtype=td, requires_grad=True)
            b = torch.randn(M, device=dev, dtype=td, requires_grad=True)
            go = torch.randn(N, S, M, device=dev, dtype=td)
            meta["op"] = "linear 768->3072 fwd+bwd"

            def op():
                torch.autograd.grad(F.linear(x, w, b), [x, w, b], go)
            return op, meta

        if name == "layernorm":
            x = torch.randn(N, S, D, device=dev, dtype=td, requires_grad=True)
            # params stay fp32, which is what nn.LayerNorm does under autocast
            w = torch.ones(D, device=dev, dtype=pd, requires_grad=True)
            b = torch.zeros(D, device=dev, dtype=pd, requires_grad=True)
            # Under autocast bf16 torch casts the layer_norm input up to fp32
            # and returns fp32 (calling it directly with bf16 x + fp32 params
            # raises);
            ac = (torch.autocast("cuda", dtype=torch.bfloat16) if dtype == "bf16"
                  else torch.autocast("cuda", enabled=False))
            with ac:
                go = torch.randn_like(F.layer_norm(x, (D,), w, b, 1e-6))
            meta["op"] = "layer_norm(768) fwd+bwd under autocast"

            def op():
                with ac:
                    o = F.layer_norm(x, (D,), w, b, 1e-6)
                torch.autograd.grad(o, [x, w, b], go)
            return op, meta

        # scaled dot-product attentoin
        hd = D // H
        q, k, v = (torch.randn(N, H, S, hd, device=dev, dtype=td,
                               requires_grad=True) for _ in range(3))
        go = torch.randn(N, H, S, hd, device=dev, dtype=td)
        meta["op"] = f"sdpa seq{S} heads{H} hd{hd} fwd+bwd"

        def op():
            torch.autograd.grad(
                F.scaled_dot_product_attention(q, k, v), [q, k, v], go)
        return op, meta

    # ---- ResNet ops -----------------------------------------------
    C, HH, WW = SHAPES[shape]
    meta.update(shape=shape, C=C, H=HH, W=WW)

    if name == "elementwise":
        x = torch.randn(N, C, HH, WW, device=dev, dtype=td)
        r = torch.randn(N, C, HH, WW, device=dev, dtype=td)
        buf = torch.empty_like(x)
        meta["op"] = "8x (add_ + relu_)"

        def op():
            y = buf.copy_(x)
            for _ in range(8):
                y.add_(r)
                torch.relu_(y)
        return op, meta

    if name == "pool":
        x = torch.randn(N, C, HH, WW, device=dev, dtype=td, requires_grad=True)
        go = torch.randn_like(F.max_pool2d(x, 3, 2, 1))
        meta["op"] = "max_pool2d k3 s2 p1 fwd+bwd"

        def op():
            torch.autograd.grad(F.max_pool2d(x, 3, 2, 1), [x], go)
        return op, meta

    if name in ("bn_fwd", "bn_bwd"):
        # what a torchvision ResNet does under autocast bf16
        need_grad = (name == "bn_bwd")
        x = torch.randn(N, C, HH, WW, device=dev, dtype=td,
                        requires_grad=need_grad)
        w = torch.ones(C, device=dev, dtype=pd, requires_grad=need_grad)
        b = torch.zeros(C, device=dev, dtype=pd, requires_grad=need_grad)
        rm = torch.zeros(C, device=dev, dtype=pd)
        rv = torch.ones(C, device=dev, dtype=pd)

        if name == "bn_fwd":
            meta["op"] = "batch_norm train=True, no_grad"

            def op():
                with torch.no_grad():
                    F.batch_norm(x, rm, rv, w, b, True, 0.1, 1e-5)
            return op, meta

        out = F.batch_norm(x, rm, rv, w, b, True, 0.1, 1e-5)
        go = torch.randn_like(out)
        meta["op"] = "batch_norm backward only (retain_graph)"

        def op():
            torch.autograd.grad(out, [x, w, b], go, retain_graph=True)
        return op, meta

    if name in CONV_OPS:
        x = torch.randn(N, C, HH, WW, device=dev, dtype=td)
        w = torch.randn(C, C, 3, 3, device=dev, dtype=td)
        if name == "conv_bwd":
            x.requires_grad_(True)
            w.requires_grad_(True)
        meta["op"] = f"conv2d {C}->{C} k3 s1 p1"

        if name == "conv_fwd":
            def op():
                with torch.no_grad():
                    F.conv2d(x, w, None, 1, 1)
            return op, meta

        out = F.conv2d(x, w, None, 1, 1)
        go = torch.randn_like(out)
        meta["op"] += " dgrad+wgrad only (retain_graph)"
        def op():
            torch.autograd.grad(out, [x, w], go, retain_graph=True)
        return op, meta

    raise SystemExit(f"microbench: unknown op {name}")


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True, choices=list(OPS),
                    help="the op to loop")
    ap.add_argument("--shape", default="s2", choices=sorted(SHAPES) + ["vit"])
    ap.add_argument("--dtype", required=True, choices=list(DTYPES))
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seconds", type=float, default=60.0,
                    help="sustained GPU-busy window AFTER --discard")
    ap.add_argument("--discard", type=float, default=15.0,
                    help="leading seconds thrown away: clock and temperature "
                         "have not settled yet")
    ap.add_argument("--target-iter-ms", type=float, default=100.0)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--axis", default="none", choices=["clk", "cap", "none"])
    ap.add_argument("--axisval", type=float, default=-1)
    ap.add_argument("--nvtx", action="store_true")
    ap.add_argument("--hz", type=float, default=20.0)
    a = ap.parse_args(argv)

    if a.op in VIT_OPS:
        a.shape = "vit"
    elif a.shape == "vit":
        sys.exit(f"--shape vit is only for {VIT_OPS}")
    return a


def time_op(fn, reps=5):
    import torch
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    t0 = MONO()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (MONO() - t0) * 1e-9 / reps


def main():
    a = parse_args()
    import torch

    os.makedirs(a.outdir, exist_ok=True)
    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)

    if a.dtype in ("fp32", "tf32"):
        want = (a.dtype == "tf32")
        torch.backends.cuda.matmul.allow_tf32 = want
        torch.backends.cudnn.allow_tf32 = want
        torch.set_float32_matmul_precision("high" if want else "highest")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)

    pynvml.nvmlInit()
    phys = visible_physical_indices()[:1]
    bus = bus_id_of(0)
    h = handle_for(phys[0], bus)[0]
    gpu = phys[0]
    sampler = Sampler(h, os.path.join(a.outdir, f"timeline_gpu{gpu}.csv"), a.hz)

    N = a.batch
    op, meta = build_op(a.op, a.shape, a.dtype, N, dev)
    op()
    torch.cuda.synchronize()

    t_op = time_op(op)
    # R: repeats per iteration.
    R = max(1, int(round(a.target_iter_ms * 1e-3 / max(t_op, 1e-6))))

    nvtx, use_nvtx = torch.cuda.nvtx, a.nvtx
    it_f = open(os.path.join(a.outdir, f"iters_gpu{gpu}.csv"), "w")
    it_f.write("iter,t_start_ns,t_end_ns,retained\n")

    def iteration(i, retained):
        # Only retained iterations get the "iter" range: discarded ones run
        # before the clock and temperature have settled.
        if use_nvtx and retained:
            nvtx.range_push("iter")
        s = MONO()
        for _ in range(R):
            op()
        torch.cuda.synchronize()
        e = MONO()
        if use_nvtx and retained:
            nvtx.range_pop()
        it_f.write(f"{i},{s},{e},{int(retained)}\n")
        return s, e

    sampler.start()

    # ---- discard window: clock and temperature have not settled yet -----
    t_warm0 = MONO()
    i = 0
    while (MONO() - t_warm0) * 1e-9 < a.discard:
        iteration(-1 - i, False)
        i += 1
    n_discarded = i

    # ---- retained window (wall-clock bounded, not iteration-count) ------
    durs = []
    t_meas0 = MONO()
    # hardware energy counter (mJ) at window open -- read alongside the
    # trapezoid so the two methods cross-check each other
    E_ctr0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
    n = 0
    while (MONO() - t_meas0) * 1e-9 < a.seconds:
        s_, e_ = iteration(n, True)
        durs.append((e_ - s_) * 1e-9)
        n += 1
    t_meas1 = MONO()
    E_ctr1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)

    sampler.stop.set()
    sampler.join(timeout=5)
    it_f.close()

    power, clocks, reasons = read_timeline(
        os.path.join(a.outdir, f"timeline_gpu{gpu}.csv"), t_meas0, t_meas1)
    E, gaps = trapezoid_energy(power, t_meas0, t_meas1)
    pin = [v for t, v in power if t_meas0 <= t <= t_meas1]
    window_s = (t_meas1 - t_meas0) * 1e-9

    summary = {
        "op": a.op, "shape": a.shape, "dtype": a.dtype,
        "N": N, "R": R, "meta": meta,
        "axis": a.axis, "axisval": a.axisval,
        "gpu": gpu,
        "window_s": window_s,
        "iters": n, "iters_discarded": n_discarded,
        "t_iter_s": statistics.median(durs) if durs else float("nan"),
        "t_op_s": (statistics.median(durs) / R) if durs else float("nan"),
        "P_med_W": statistics.median(pin) if pin else None,
        "E_J": E,
        "E_counter_J": (E_ctr1 - E_ctr0) / 1000.0,
        "E_J_per_op": (E / (n * R)) if n else float("nan"),
        "clk_med_MHz": statistics.median(clocks) if clocks else None,
        "reasons_seen": sorted(set(reasons)),
        "throttle_frac": (sum(1 for r in reasons if r & 0x4) / len(reasons)
                          if reasons else None),
        "energy_gaps_skipped": gaps,
        # retained-window bounds
        "t_meas0_ns": t_meas0, "t_meas1_ns": t_meas1,
        "host": os.uname().nodename,
    }
    with open(os.path.join(a.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("SUMMARY " + json.dumps(
        {k: summary[k] for k in ("op", "shape", "dtype", "N", "R", "axis",
                                 "axisval", "iters", "t_op_s", "P_med_W",
                                 "E_J", "E_counter_J", "clk_med_MHz")}),
          flush=True)
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
