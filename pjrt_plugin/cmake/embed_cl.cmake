# Concatenates the ordered OpenCL sources in ${SRCS} (semicolon-separated) into
# ${OUT} as `static const char kVmClSource[]` (one raw string literal). The
# files compile as one translation unit (clCreateProgramWithSource), giving
# per-op-family file modularity with functions inlined — no clLinkProgram.
# Order matters: vm_common first (defines/structs), then ops/*, then vm_main
# (exec_tiles dispatch + kernel).
string(REPLACE "," ";" SRC_LIST "${SRCS}")   # undo the | join from CMakeLists
file(WRITE ${OUT} "// Generated from kernels/*.cl — do not edit.\n"
                  "static const char kVmClSource[] = R\"CLSRC(\n")
foreach(f ${SRC_LIST})
  file(READ ${f} PART)
  file(APPEND ${OUT} "${PART}\n")
endforeach()
file(APPEND ${OUT} ")CLSRC\";\n")
