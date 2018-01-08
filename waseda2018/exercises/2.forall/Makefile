NVCC=nvcc
ARCH=sm_37

all: forall

forall: forall.cu
	$(NVCC) -O3 -o $@ -arch $(ARCH) $^

.PHONY: clean
clean:
	rm forall
