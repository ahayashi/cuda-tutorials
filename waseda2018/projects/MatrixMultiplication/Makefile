NVCC=nvcc
ARCH=sm_37

all: cpu gpu

cpu: mm.c
	gcc -O3 -o $@ $^

gpu: mm.cu
	$(NVCC) -O3 -o $@ -arch=$(ARCH) $^

.PHONY: clean
clean:
	rm cpu gpu
