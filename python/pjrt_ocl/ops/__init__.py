"""Op-family modules. Importing this package registers every family's
stablehlo handlers (into lowering.OP_HANDLERS) and tensor-opcode semantics
(into opsem.TO_TASK/INTERP/READS).

Each family is a standalone module so coverage can grow in parallel without
editing the core files. Add a new family by creating pjrt_ocl/ops/<name>.py
and importing it here.
"""
from . import shape  # noqa: F401  (broadcast_in_dim, transpose, reshape)
from . import making  # noqa: F401  (iota, convert)
