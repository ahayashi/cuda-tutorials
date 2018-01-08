NVCC=nvcc
ARCH=sm_37

all: time

time: time.cu
	$(NVCC) -O3 -o $@ -arch $(ARCH) $^

.PHONY: clean
clean:
	rm time
