CC = cc
CFLAGS = -O2 -Wno-deprecated-declarations
LIBS = -lOpenCL -lm

all: probe bench17 bench17hp bench17mma bench17f16in

probe: probe.c probe.cl
	$(CC) $(CFLAGS) -o probe probe.c $(LIBS)

bench17: bench17.c mma17.cl
	$(CC) $(CFLAGS) -o bench17 bench17.c $(LIBS)

# half-precision (fp16/bf16) tensor-core ceiling — §38, breaks the tf32 57 wall
bench17hp: bench17hp.c mma17_hp.cl
	$(CC) $(CFLAGS) -o bench17hp bench17hp.c $(LIBS)

# §39: modern mma.sync.m16n8k16 + ldmatrix (f32 arena). Ties wmma at ~92 —
# proves the ceiling is staging-bound, not fragment-register-bound.
bench17mma: bench17mma.c mma17_mma.cl
	$(CC) $(CFLAGS) -o bench17mma bench17mma.c $(LIBS)

# §39: f16-INPUT staging (half the global bytes). Breaks 92 -> ~107 TF/s.
bench17f16in: bench17f16in.c mma17_f16in.cl
	$(CC) $(CFLAGS) -o bench17f16in bench17f16in.c $(LIBS)

# cp.async feasibility gate (must run in this dir; reads probe.cl)
run-probe: probe
	./probe

# standalone tensor-core matmul ceiling sweep (tf32)
run-bench: bench17
	./bench17

# fp16/bf16 tensor-core ceiling sweep (§38 — ~92 TF/s, beats tf32 57)
run-bench-hp: bench17hp
	./bench17hp

# §39 mma.sync + ldmatrix (f32 arena; ties wmma ~92, confirms staging-bound)
run-bench-mma: bench17mma
	CLSRC=mma17_mma.cl ./bench17mma

# §39 f16-input staging (breaks 92 -> ~107 TF/s @4096; gap to cuBLAS 1.46->1.25x)
run-bench-f16in: bench17f16in
	./bench17f16in

clean:
	rm -f probe bench17 bench17hp bench17mma bench17f16in
