NVCC=nvcc
ARCH=sm_30

all: reductionAtomic reductionWarpShuffleAtomic reductionBlockShuffleAtomic reductionBlockShuffle reductionCUB

reductionAtomic: reductionAtomic.cu
	$(NVCC) -O3 -o $@ -arch $(ARCH) $^

reductionWarpShuffleAtomic: reductionWarpShuffleAtomic.cu
	$(NVCC) -O3 -o $@ -arch $(ARCH) $^

reductionBlockShuffleAtomic: reductionBlockShuffleAtomic.cu
	$(NVCC) -O3 -o $@ -arch $(ARCH) $^

reductionBlockShuffle: reductionBlockShuffle.cu
	$(NVCC) -O3 -o $@ -arch $(ARCH) $^

reductionCUB: reductionCUB.cu
	$(NVCC) -O3 -o $@ -arch $(ARCH) $^

.PHONY: clean
clean:
	rm reductionAtomic reductionWarpShuffleAtomic reductionBlockShuffleAtomic reductionBlockShuffle reductionCUB
