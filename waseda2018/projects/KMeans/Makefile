NVCC=nvcc
ARCH=sm_37

all: cpu gpu

cpu: kmeans.c
	gcc -O3 -o $@ $^

gpu: kmeans.cu
	$(NVCC) -O3 -o $@ -arch=$(ARCH) $^

.PHONY: clean
clean:
	rm cpu gpu
