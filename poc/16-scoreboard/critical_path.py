"""Honest overlap ceiling for the base transformer (§14a check before rewiring).

A scoreboard can only overlap work that is truly INDEPENDENT. The ceiling is:
    speedup_max = total_phase_cost / critical_path_cost
where the critical path is the longest DEPENDENCY chain through the phase DAG.
If critical_path ~= total, the program is a dependency chain and a scoreboard
buys ~nothing on it — STOP (§14a).

Cost model: every barrier phase is latency-bound ~1 tile deep on 376 lanes (§29:
skew ~= wall, phases <=376 tiles run ~1 tile time). So weight each phase by the
per-tile cost of its heaviest op class (a MMA tile >> an EW tile). This mirrors
the measured 5.0 ms budget (matmul 84%)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python"))
import jax, numpy as np, importlib.util, collections
spec = importlib.util.spec_from_file_location("bt", os.path.join(os.path.dirname(__file__), "../../tools/bench_transformer.py"))
bt = importlib.util.module_from_spec(spec); spec.loader.exec_module(bt)
from jaxlib.mlir.dialects import stablehlo
from pjrt_ocl import lowering as L, scheduler as S

CFG = os.environ.get("CFG", "base")
print(f"### config = {CFG}")
x, params = bt.make_params(bt.CONFIGS[CFG])
H = bt.CONFIGS[CFG][3]
fn = jax.jit(lambda x, p: model(x, p, H)) if False else None
model = bt.make_model(jax.numpy)
fn = jax.jit(lambda x, p: model(x, p, H))
art = fn.lower(x, params).compiler_ir("stablehlo")
with art.context:
    artifact = stablehlo.serialize_portable_artifact(art, stablehlo.get_current_version())
prog = L.lower_artifact(artifact)
sc = S._Scheduler(prog, S.DeviceConfig(nlanes=376), 376)
levels = sc._build_levels(list(range(prog.main_len)))

# per-tile cost weights (relative; MMA tile ~ the §29 straggler). Tuned so the
# MMA share ~= 84% of total, matching §29.
def tile_cost(op):
    return {S.TILE_MMA: 30.0}.get(op, 1.0)

# phase cost = heaviest op-class tile cost present (latency-bound, 1 tile deep)
phase_cost = []
phase_writes = []   # buffers written by each phase
phase_reads = []
for kind, payload in levels:
    idxs = payload if kind == "compute" else [payload]
    cost = 0.0
    w, r = set(), set()
    for i in idxs:
        t = sc.tasks[sc._task_for(i)] if kind == "compute" else None
        c = tile_cost(t.tile_op) if t is not None else 5.0
        cost = max(cost, c)
        w |= S._writes(prog.instrs[i]); r |= S._reads(prog.instrs[i])
    phase_cost.append(cost); phase_writes.append(w); phase_reads.append(r)

N = len(levels)
# phase DAG: phase b depends on phase a<b if b reads/overwrites what a wrote (RAW/WAW)
# or a reads what b writes (WAR). Build direct edges.
producers = {}   # buffer -> last phase that wrote it (for RAW/WAW) — but keep all
deps = [set() for _ in range(N)]
last_writer = {}
readers_since = collections.defaultdict(list)
for b in range(N):
    for buf in phase_reads[b]:
        if buf in last_writer:
            deps[b].add(last_writer[buf])          # RAW
    for buf in phase_writes[b]:
        if buf in last_writer:
            deps[b].add(last_writer[buf])          # WAW
        for rp in readers_since.get(buf, []):
            if rp != b:
                deps[b].add(rp)                    # WAR
    for buf in phase_writes[b]:
        last_writer[buf] = b
        readers_since[buf] = []
    for buf in phase_reads[b]:
        readers_since[buf].append(b)

# longest weighted path (critical path) over the DAG
cp = [0.0] * N
for b in range(N):
    base = max((cp[a] for a in deps[b]), default=0.0)
    cp[b] = base + phase_cost[b]
critical = max(cp) if N else 0.0
total = sum(phase_cost)
print(f"phases={N}  total_phase_cost={total:.0f}  critical_path={critical:.0f}")
print(f"MAX overlap speedup ceiling (total/critical) = {total/critical:.3f}x")
print(f"  => a perfect scoreboard could remove at most "
      f"{100*(1-critical/total):.1f}% of the serial phase cost")

# OPTIMISTIC ceiling: RAW-only edges (minimal constraints = MAX overlap possible)
raw = [set() for _ in range(N)]
lw = {}
for b in range(N):
    for buf in phase_reads[b]:
        if buf in lw:
            raw[b].add(lw[buf])
    for buf in phase_writes[b]:
        lw[buf] = b
cpr = [0.0]*N
for b in range(N):
    base = max((cpr[a] for a in raw[b]), default=0.0)
    cpr[b] = base + phase_cost[b]
crit_raw = max(cpr) if N else 0.0
print(f"\nRAW-ONLY (optimistic) critical_path={crit_raw:.0f} of {total:.0f} "
      f"=> ceiling {total/crit_raw:.3f}x ({100*(1-crit_raw/total):.1f}% overlappable)")

# same, MMA-only (the 84% bucket): how much MMA work is off the critical path?
mma_total = sum(c for c in phase_cost if c > 5.0)
# critical path restricted to MMA cost contribution
cp2 = [0.0]*N
for b in range(N):
    base = max((cp2[a] for a in deps[b]), default=0.0)
    cp2[b] = base + (phase_cost[b] if phase_cost[b] > 5.0 else 0.0)
mma_crit = max(cp2) if N else 0
print(f"\nMMA phase cost total={mma_total:.0f}  on critical path={mma_crit:.0f}")
print(f"  MMA off critical path (overlappable) = {100*(1-mma_crit/mma_total):.1f}%")
