# poc/15 register-budget probe. Needs the generated vm_cl_source.h from a dev
# build of the plugin (cmake -S pjrt_plugin -B pjrt_plugin/build-dev ...).
HDR ?= ../../pjrt_plugin/build-dev
regprobe: regprobe.c $(HDR)/vm_cl_source.h
	cc -O2 -I $(HDR) regprobe.c -lOpenCL -o regprobe
clean:
	rm -f regprobe
.PHONY: clean
