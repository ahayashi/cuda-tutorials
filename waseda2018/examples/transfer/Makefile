NVCC=nvcc
ARCH=sm_37

all: transfer

transfer: transfer.cu
	$(NVCC) -O3 -o $@ -arch $(ARCH) $^

.PHONY: clean
clean:
	rm transfer
