CC = cc
CFLAGS = -O2 -Wno-deprecated-declarations
LIBS = -lOpenCL -lm

all: probe bench17 bench17hp

probe: probe.c probe.cl
	$(CC) $(CFLAGS) -o probe probe.c $(LIBS)

bench17: bench17.c mma17.cl
	$(CC) $(CFLAGS) -o bench17 bench17.c $(LIBS)

# half-precision (fp16/bf16) tensor-core ceiling — §38, breaks the tf32 57 wall
bench17hp: bench17hp.c mma17_hp.cl
	$(CC) $(CFLAGS) -o bench17hp bench17hp.c $(LIBS)

# cp.async feasibility gate (must run in this dir; reads probe.cl)
run-probe: probe
	./probe

# standalone tensor-core matmul ceiling sweep (tf32)
run-bench: bench17
	./bench17

# fp16/bf16 tensor-core ceiling sweep (§38 — ~92 TF/s, beats tf32 57)
run-bench-hp: bench17hp
	./bench17hp

clean:
	rm -f probe bench17 bench17hp
