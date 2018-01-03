NVCC=nvcc
ARCH=sm_30

all: hello

hello: hello.cu
	$(NVCC) -O3 -o $@ -arch $(ARCH) $^

.PHONY: clean
clean:
	rm hello
