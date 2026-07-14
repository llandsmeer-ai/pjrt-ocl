"""jax_plugins namespace-package discovery hook for the pjrt-ocl PoC.

JAX imports every module under the `jax_plugins` namespace package and calls
its `initialize()`. For the PoC this directory just needs to be on sys.path
(e.g. run python from poc/02-pjrt-skeleton/, since `python -c` puts the cwd on
sys.path). No pip packaging yet.
"""

import os

from jax._src import xla_bridge as xb

_SO = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "build",
                 "libpjrt_ocl_skeleton.so")
)


def initialize():
    if not os.path.exists(_SO):
        raise FileNotFoundError(
            f"pjrt-ocl skeleton .so not built: {_SO} "
            "(run: cmake -S . -B build -G Ninja && cmake --build build)"
        )
    if "opencl" in xb._backend_factories:  # already registered explicitly
        return
    xb.register_plugin("opencl", priority=500, library_path=_SO, options=None)
