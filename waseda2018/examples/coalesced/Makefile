NVCC=nvcc
ARCH=sm_37

all: coalesced

coalesced: coalesced.cu
	$(NVCC) -O3 -o $@ -arch $(ARCH) $^

.PHONY: clean
clean:
	rm coalesced
