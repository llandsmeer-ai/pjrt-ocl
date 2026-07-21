"""Diverse AI + scientific-computing workload testbench for the OpenCL backend.

See ``run_suite.py`` for the driver. Each workload is a ``build(name)`` case in
``workloads.py`` returning a jittable fn + a pytree of deterministic f32 inputs.
"""
