CC = cc
CFLAGS = -O2 -Wno-deprecated-declarations
LIBS = -lOpenCL -lm

all: probe bench17

probe: probe.c probe.cl
	$(CC) $(CFLAGS) -o probe probe.c $(LIBS)

bench17: bench17.c mma17.cl
	$(CC) $(CFLAGS) -o bench17 bench17.c $(LIBS)

# cp.async feasibility gate (must run in this dir; reads probe.cl)
run-probe: probe
	./probe

# standalone tensor-core matmul ceiling sweep
run-bench: bench17
	./bench17

clean:
	rm -f probe bench17
