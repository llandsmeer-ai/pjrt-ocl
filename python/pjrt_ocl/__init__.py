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


def _bundled_plugin_path() -> str:
    # A packaged build places the .so next to this file (pjrt_ocl/libpjrt_ocl.so).
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "libpjrt_ocl.so")


def _default_plugin_path() -> str:
    # This file lives at <repo>/python/pjrt_ocl/__init__.py when installed
    # editable; the plugin builds to <repo>/pjrt_plugin/build/libpjrt_ocl.so.
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(repo, "pjrt_plugin", "build", "libpjrt_ocl.so")


def find_plugin_library() -> str:
    """Absolute path of the built plugin .so, or raise FileNotFoundError.
    Search order: PJRT_OCL_PLUGIN_PATH, bundled-in-package, dev build tree."""
    env = os.environ.get(PLUGIN_PATH_ENV)
    if env:
        if not os.path.isfile(env):
            raise FileNotFoundError(
                f"{PLUGIN_PATH_ENV}={env!r} is set but that file does not exist")
        return os.path.abspath(env)
    bundled = _bundled_plugin_path()
    if os.path.isfile(bundled):
        return bundled
    default = _default_plugin_path()
    if os.path.isfile(default):
        return default
    raise FileNotFoundError(
        "pjrt-ocl native plugin (libpjrt_ocl.so) not found. A plain "
        "`pip install` does NOT build it. Either:\n"
        "  1. clone the repo and build it:\n"
        "     cmake -S pjrt_plugin -B pjrt_plugin/build -G Ninja && "
        "cmake --build pjrt_plugin/build\n"
        f"  2. then either run from the clone, or set {PLUGIN_PATH_ENV} to the "
        "built .so:\n"
        f"     export {PLUGIN_PATH_ENV}=/path/to/pjrt-ocl/pjrt_plugin/build/"
        "libpjrt_ocl.so\n"
        f"(searched: {PLUGIN_PATH_ENV} [unset], bundled {bundled}, dev "
        f"{default})")


def lower_service_path() -> str:
    """Absolute path of the lowering-service script (the C++ side execs this)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "lower_service.py")


def initialize() -> None:
    """jax_plugins hook: register the 'opencl' PJRT platform with jax."""
    try:
        library_path = find_plugin_library()
    except FileNotFoundError as e:
        # Visible (not just logging): otherwise the user only sees jax's opaque
        # "Backend 'opencl' is not in the list of known backends" downstream.
        # Only shout when the user actually asked for opencl.
        if "opencl" in os.environ.get("JAX_PLATFORMS", ""):
            sys.stderr.write("\npjrt-ocl: cannot register the 'opencl' "
                             "backend:\n" + str(e) + "\n\n")
        else:
            logger.warning("pjrt-ocl: 'opencl' backend not registered: %s", e)
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
