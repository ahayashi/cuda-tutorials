NVCC=nvcc
ARCH=sm_37

all: cpu gpu

cpu: pagerank.c
	gcc -O3 -o $@ $^ 

gpu: pagerank.cu
	$(NVCC) -O3 -o $@ -arch=$(ARCH) $^

.PHONY: clean
clean:
	rm cpu gpu
