NVCC=nvcc
ARCH=sm_70

all: reductionAtomic reductionWarpShuffleAtomic reductionBlockShuffleAtomic reductionBlockShuffle reductionCUB reductionWarpShuffleAtomicCG reductionBlockShuffleCGSync

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

reductionWarpShuffleAtomicCG: reductionWarpShuffleAtomicCG.cu
	$(NVCC) -O3 -o $@ -rdc=true -arch $(ARCH) $^
reductionBlockShuffleCGSync: reductionBlockShuffleCGSync.cu
	$(NVCC) -O3 -o $@ -rdc=true -arch $(ARCH) $^

.PHONY: clean
clean:
	rm reductionAtomic reductionWarpShuffleAtomic reductionBlockShuffleAtomic reductionBlockShuffle reductionCUB reductionWarpShuffleAtomicCG reductionBlockShuffleCGSync
