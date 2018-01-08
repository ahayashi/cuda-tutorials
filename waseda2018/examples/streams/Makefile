NVCC=nvcc
ARCH=sm_37

all: streams

streams: streams.cu
	$(NVCC) -O3 -o $@ -arch $(ARCH) $^

.PHONY: clean
clean:
	rm streams
