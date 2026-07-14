# Embeds ${IN} into ${OUT} as `static const char kVmClSource[]` via a raw
# string literal (delimiter chosen to never appear in OpenCL C source).
file(READ ${IN} SRC)
file(WRITE ${OUT}
  "// Generated from kernels/vm.cl — do not edit.\n"
  "static const char kVmClSource[] = R\"CLSRC(\n${SRC})CLSRC\";\n")
