"""Inspect the REAL base-transformer schedule: how many phases, and how much
INDEPENDENT shaped (matmul) work sits in separate phases that a scoreboard could
overlap. Confirms the §29/§30 premise before rewiring the scheduler (§14a)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python"))
import jax, numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location("bt", os.path.join(os.path.dirname(__file__), "../../tools/bench_transformer.py"))
bt = importlib.util.module_from_spec(spec); spec.loader.exec_module(bt)
from jaxlib.mlir.dialects import stablehlo
from pjrt_ocl import lowering as L, scheduler as S

cfg = bt.CONFIGS["base"]
model = bt.make_model(jax.numpy)
x, params = bt.make_params(cfg)
B, T, D, H, F, Ln = cfg
fn = jax.jit(lambda x, p: model(x, p, H))
art = fn.lower(x, params).compiler_ir("stablehlo")
with art.context:
    artifact = stablehlo.serialize_portable_artifact(art, stablehlo.get_current_version())

prog = L.lower_artifact(artifact)
cfgd = S.DeviceConfig(nlanes=376)
sc = S._Scheduler(prog, cfgd, 376)

# reproduce the root levels the scheduler builds
levels = sc._build_levels(list(range(prog.main_len)))
n_compute = sum(1 for k, _ in levels if k == "compute")
n_region = sum(1 for k, _ in levels if k != "compute")
print(f"main_len={prog.main_len} instrs;  root levels: {len(levels)} "
      f"({n_compute} compute phases, {n_region} region ops)")

MMA = S.TILE_MMA
shaped_phases = 0
multi_shaped_phases = 0
lone_matmul_phases = 0
total_mma_tiles = 0
mma_tiles_per_phase = []
for kind, payload in levels:
    if kind != "compute":
        continue
    shaped = [i for i in payload if not sc._is_map(i)]
    mma = [i for i in shaped if sc.tasks[sc._task_for(i)].tile_op == MMA]
    if mma:
        shaped_phases += 1
        tiles = sum(sc.tasks[sc._task_for(i)].n_tiles() for i in mma)
        total_mma_tiles += tiles
        mma_tiles_per_phase.append((len(mma), tiles))
        if len(mma) >= 2:
            multi_shaped_phases += 1
        else:
            lone_matmul_phases += 1

print(f"\nMMA phases: {shaped_phases}  "
      f"(lone-matmul: {lone_matmul_phases}, multi-matmul: {multi_shaped_phases})")
print(f"total MMA tiles across program: {total_mma_tiles}")
print(f"mean MMA-tile occupancy per phase / 376 lanes: "
      f"{np.mean([t for _, t in mma_tiles_per_phase]) / 376:.3f}")
print("\nper-phase (n_matmuls, total_mma_tiles, occupancy):")
for nm, t in mma_tiles_per_phase:
    print(f"  matmuls={nm:2d}  tiles={t:4d}  occ={t/376:.2f}")

# The key question: are consecutive lone-matmul phases INDEPENDENT of each other
# (a scoreboard could run them concurrently)? Check dep between adjacent MMA phases.
print("\n--- independence of consecutive MMA phases (could a scoreboard overlap them?) ---")
mma_phase_payloads = [p for k, p in levels if k == "compute"
                      and any(sc.tasks[sc._task_for(i)].tile_op == MMA for i in p)]
def phase_dep(later, earlier):
    return any(S._depends(prog.instrs, j, i) for j in later for i in earlier)
indep_pairs = 0
for a in range(1, len(mma_phase_payloads)):
    if not phase_dep(mma_phase_payloads[a], mma_phase_payloads[a-1]):
        indep_pairs += 1
print(f"adjacent MMA-phase pairs that are INDEPENDENT: {indep_pairs}/{len(mma_phase_payloads)-1}")
