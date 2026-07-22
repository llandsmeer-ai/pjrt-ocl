CFLAGS ?= -O2 -Wall
LDLIBS  = -lOpenCL -lm

all: probe21 bench21 tf32bits

probe21: probe21.c
bench21: bench21.c
tf32bits: tf32bits.c

clean:
	rm -f probe21 bench21 tf32bits

.PHONY: all clean
