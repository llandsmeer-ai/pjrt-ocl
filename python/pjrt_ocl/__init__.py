"""pjrt-ocl: a PJRT plugin that lets JAX execute on OpenCL devices.

JAX discovers this package via the `jax_plugins` entry point (see pyproject.toml)
and calls `initialize()` at backend-discovery time. `initialize()` registers the
C++ PJRT plugin .so under platform name 'opencl' and passes it the info it needs
to spawn the Python lowering subprocess (see docs/decisions.md #2):

    options = {
        'python_exe':    absolute path of this interpreter (sys.executable),
        'lower_service': absolute path of pjrt_ocl/lower_service.py,
    }

The C++ plugin runs `<python_exe> <lower_service>` at PJRT_Client_Compile time,
piping the VHLO portable artifact to stdin and reading VMProgram v1 bytes from
stdout (docs/vmprogram.md).

The plugin .so is located via, in order:
  1. env var PJRT_OCL_PLUGIN_PATH (must exist if set)
  2. <repo>/pjrt_plugin/build/libpjrt_ocl.so (editable-install source layout)

If neither exists, `initialize()` logs a warning and skips registration instead
of raising — a source checkout without a built plugin must not break `import jax`
for every other backend. (jax's discover_pjrt_plugins() would swallow the
exception anyway, but with a full logged traceback; skipping is cleaner.)
"""
from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)

PLUGIN_PATH_ENV = "PJRT_OCL_PLUGIN_PATH"


def _default_plugin_path() -> str:
    # This file lives at <repo>/python/pjrt_ocl/__init__.py when installed
    # editable; the plugin builds to <repo>/pjrt_plugin/build/libpjrt_ocl.so.
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(repo, "pjrt_plugin", "build", "libpjrt_ocl.so")


def find_plugin_library() -> str:
    """Absolute path of the built plugin .so, or raise FileNotFoundError."""
    env = os.environ.get(PLUGIN_PATH_ENV)
    if env:
        if not os.path.isfile(env):
            raise FileNotFoundError(
                f"{PLUGIN_PATH_ENV}={env!r} is set but that file does not exist")
        return os.path.abspath(env)
    default = _default_plugin_path()
    if os.path.isfile(default):
        return default
    raise FileNotFoundError(
        f"pjrt-ocl plugin library not found: {PLUGIN_PATH_ENV} is unset and the "
        f"default build location {default} does not exist. Build the plugin "
        f"(cmake -S pjrt_plugin -B pjrt_plugin/build -G Ninja && "
        f"cmake --build pjrt_plugin/build) or set {PLUGIN_PATH_ENV} to a built "
        f"libpjrt_ocl.so.")


def lower_service_path() -> str:
    """Absolute path of the lowering-service script (the C++ side execs this)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "lower_service.py")


def initialize() -> None:
    """jax_plugins hook: register the 'opencl' PJRT platform with jax."""
    try:
        library_path = find_plugin_library()
    except FileNotFoundError as e:
        logger.warning(
            "pjrt-ocl: skipping registration of the 'opencl' PJRT platform "
            "(plugin not built?): %s", e)
        return

    # Import lazily: lower_service.py imports this package too and must stay
    # light (no jax import) in the compile-time subprocess.
    from jax._src import xla_bridge as xb

    if "opencl" in xb._backend_factories:  # already registered (e.g. manually)
        return

    options = {
        "python_exe": sys.executable,
        "lower_service": lower_service_path(),
    }
    xb.register_plugin("opencl", priority=500, library_path=library_path,
                       options=options)
