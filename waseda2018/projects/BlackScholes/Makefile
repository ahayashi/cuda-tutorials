NVCC=nvcc
ARCH=sm_37

all: cpu gpu

cpu: bs.c
	gcc -O3 -o $@ $^ -lm

gpu: bs.cu
	$(NVCC) -O3 -o $@ -arch=$(ARCH) $^

.PHONY: clean
clean:
	rm cpu gpu
