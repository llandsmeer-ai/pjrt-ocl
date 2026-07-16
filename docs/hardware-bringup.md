# New-hardware bring-up checklist

Run this end-to-end **every time a new OpenCL device/vendor/driver first meets this
codebase** (new GPU, new CPU runtime, big driver update). Each step says what to run,
what "good" looks like, and where to record the outcome. Findings that changed a
decision go to `docs/decisions.md` (§9 = the Intel Xe2 bring-up, is the worked
example); plain results go to the README hardware section and `tests/SCOREBOARD.md`.

Do the steps in order — each one assumes the previous ones pass.

## 0. Environment & device enumeration

- `clinfo -l` must list the device. If it doesn't, fix the ICD first and **record
  how** in decisions.md §8 (Environment): vendors need different things
  (NVIDIA: manual `/etc/OpenCL/vendors/nvidia.icd`; Intel Lunar Lake: needs
  `intel-opencl-icd` ≥ 24.x from `ppa:kobuk-team/intel-graphics`, the 24.04 archive
  version silently enumerates nothing; containers: `/dev/dri` must be passed through —
  `EPERM` on open even as root means the device cgroup blocks it, fix is host-side
  `docker run --device /dev/dri`).
- Record: platform/device name string, driver version (`clinfo | grep Driver`),
  "CU" count **and what a CU means on this vendor** (NVIDIA: SM; Intel: XVE;
  PoCL: core — this burned us once, see decisions §9).
- `. ./env.sh` before everything, always.

## 1. Smoke: plugin init + dialect probe

```
JAX_PLATFORMS=opencl PJRT_OCL_DEVICE=<substr> .venv/bin/python -c "import jax; print(jax.devices()[0].device_kind)"
PJRT_OCL_DEVICE=<substr> ./pjrt_plugin/build/runtime_test
```

- Init must succeed and pick a sane `-cl-std` build variant (strict compilers reject
  the device-fence builtins under the 1.2 default — decisions §3b). If the build fell
  through to `VMO_NO_DEVICE_FENCE`, the megakernel is disabled by design; note it.
- `runtime_test` header prints **"(N lanes advertised)"** — that's the next step's input.

## 2. Lane discovery (the `nlanes==0` probe) — does it hold here?

The runtime measures co-resident workgroups at init by launching the real `vm2`
kernel in probe mode (`ProbeResidency`, poc/08) and takes `min(measured, 2×CU)`.
On each new device, verify the measurement is *true*, not just plausible:

- Sanity vs vendor math: does N match HW threads ÷ (256 / kernel SIMD)? Get the
  kernel's compiled SIMD from `CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE`
  (`poc/08-occupancy-discovery/` prints all of this — run it).
- **Tightness**: `PJRT_OCL_VM_LANES=<N>` full pytest must pass;
  `PJRT_OCL_VM_LANES=<N+1>` e2e (`tests/_e2e_body.py`) should fail/hang
  (run under `timeout`, expect to kill it; the driver must recover — check with a
  normal run afterwards). If N+k works for large k, the device likely preempts
  (poc/08 found Xe2 preempts slim kernels) — then co-residency isn't binding and
  the 2×CU cap is the limiter; consider lifting it for this device.
- If the probe returns 0 or garbage: the runtime silently falls back to 2×CU —
  that's exactly the config that deadlocked Xe2, so debug the probe, don't ship the
  fallback. Record the discovered N in decisions.md.

## 3. Engine choice: megakernel vs host-dispatch

Run the full testbench under **both** engines:

```
JAX_PLATFORMS=opencl PJRT_OCL_DEVICE=<substr> .venv/bin/python -m pytest tests/ -q
JAX_PLATFORMS=opencl PJRT_OCL_DEVICE=<substr> PJRT_OCL_ENGINE=host .venv/bin/python -m pytest tests/ -q
```

- Both green is the goal (Xe2 and PoCL both are). If megakernel is flaky here and
  host-dispatch is clean, force host for this device class in `runtime.cc` engine
  selection and write down why (that rule already exists for CPUs and for
  fence-less builds).
- Also compare their *performance* in step 4 (`plot_bench.py --engine mega|host`) —
  the default engine for the device should be the faster correct one.

## 4. Performance test — vs a native plugin if one exists

```
. ./env.sh && .venv/bin/python tools/plot_bench.py --device <substr> --out docs/bench_plot_<name>.png
```

- Reference backend: native JAX plugin on the *same* silicon when it exists
  (CUDA jaxlib on NVIDIA — apples-to-apples), else JAX CPU (honest but
  cross-device; say so in the README). `--ref cuda|cpu` overrides.
- What counts as **"bad performance"** (investigate, don't just record):
  - a *cliff* in the N-vs-time curve (e.g. the 512K-element step on NVIDIA —
    lane/tile scaling threshold);
  - large-N elementwise more than ~10x off the reference **or** well under the
    device's theoretical bandwidth (compute GB/s = 3×4×N/t for `add`);
  - `while` per-iteration overhead ≫ the ~µs barrier cost measured by poc/08
    liveness (means control-flow round-trips dominate);
  - matmul way off any tuned library — expected for the naive tile kernel, but
    track the ratio so regressions show.
- Per-op lane scaling: `tools/bench_ops.sh` (edit the device/lane list for the new
  hardware; lanes ∈ {1, N/4, N/2, N} of the discovered count).
- Fix what's fixable now, file the rest as decisions.md entries with numbers.

## 5. Scheduler / cost model check

The scheduler packs lanes using per-tile-op costs (µs/tile) that are
**auto-calibrated on first run** (`CalibrateCosts` µbenchmark at init; cached
JSON keyed by platform+device+driver under `~/.cache/pjrt-ocl/`, overridable
via `PJRT_OCL_CACHE_DIR`). On a new device, verify rather than assume:

- Run once with `PJRT_OCL_LOG=1` and confirm the "cost table" line reports
  measured-or-cached, not "calibration failed"; sanity-check the JSON ratios
  against the device (e.g. an MMA tile is ~25x an EW tile on PoCL, near parity
  on a big GPU). `PJRT_OCL_CALIBRATE=1` forces re-measurement past the cache;
  a hand-written `PJRT_OCL_COST_TABLE` supersedes calibration for experiments.
- Bubble check: trace mode — run a representative program with
  `PJRT_OCL_VM_TRACE=<file>` and plot with `tools/plot_schedule.py`
  (scheduled-vs-measured lane timelines; forces the host-dispatch engine, so
  it's a timeline, not a benchmark). Lanes idling at barriers = the cost model
  or packer mis-estimates on this device — recalibrate, re-trace, and if the
  gap persists file a decisions.md entry with the plot.

## 6. Write it down (nothing is done until this is)

- `docs/decisions.md`: one bring-up section (like §9) — environment, discovered
  lanes, engine choice, surprises, perf anomalies with numbers.
- `tests/SCOREBOARD.md`: add the device to the validated-hardware line.
- `README.md` → **"Hardware tested & benchmarks"**: add a subsection for the
  device — driver/ICD used, testbench result, the bench plot, and 2–3 honest
  takeaways so users can judge whether the library is worth it on that hardware.
- If it's a new dev *host* (not just device): env facts → CLAUDE.md; host quirks
  and repro commands → session memory / a git-hidden note.
