"""Op-family modules. Importing this package registers every family's
stablehlo handlers (into lowering.OP_HANDLERS) and tensor-opcode semantics
(into opsem.TO_TASK/INTERP/READS).

Each family is a standalone module so coverage can grow in parallel without
editing the core files. Add a new family by creating pjrt_ocl/ops/<name>.py
and importing it here.
"""
from . import shape  # noqa: F401  (broadcast_in_dim, transpose, reshape)
from . import elementwise  # noqa: F401  (div/max/min/pow, unary, compare, select)
from . import dot  # noqa: F401  (dot_general — plain 2D matmul via TILE_MMA)
from . import reduce  # noqa: F401  (reduce: full sum/max/min/prod)
from . import making  # noqa: F401  (iota, convert)
from . import bitcast  # noqa: F401  (bitcast_convert — bit reinterpret)
from . import concat  # noqa: F401  (concatenate/pad via scatter)
from . import dynslice  # noqa: F401  (dynamic_slice/update via runtime-offset gather/scatter)
from . import reduce_window  # noqa: F401  (pooling via windowed reduction)
from . import region  # noqa: F401  (register-resident fused map-region, §27/§28)
from . import attention  # noqa: F401  (fused flash-attention via online softmax, §34)
