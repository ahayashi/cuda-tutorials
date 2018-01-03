NVCC=nvcc
ARCH=sm_30

all: vecadd

vecadd: vecadd.cu
	$(NVCC) -O3 -o $@ -arch $(ARCH) $^

.PHONY: clean
clean:
	rm vecadd
