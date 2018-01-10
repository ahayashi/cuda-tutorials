NVCC=nvcc
ARCH=sm_37

all: streams

streams: streams.cu
	$(NVCC) --default-stream per-thread -O3 -o $@ -arch $(ARCH) $^

.PHONY: clean
clean:
	rm streams
