#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>

long long getCurrentTime() {
    struct timeval te;
    gettimeofday(&te, NULL); // get current time
    long long microseconds = te.tv_sec*1000000LL + te.tv_usec; 
    return microseconds;
}

#define CUDA_ERROR_CHECK
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
    #ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
	fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
		 file, line, cudaGetErrorString( err ) );
	exit( -1 );
    }
    #endif

    return;
}

__global__ void coalesced(int *A, int *B, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
	// Coalesced reads and writes
	A[i] = B[i];
    }
}

__global__ void stride(int *A, int *B, int N)
{
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 32;
    if (i < N) {
	// Stride reads and writes
	A[i] = B[i];
    }
}

int main() {
    int *A, *B;
    int *dA, *dB;
    int N =  64 * 1024 * 1024;

    // Allocate memory on the host
    A = (int*)malloc(sizeof(int) * N);
    B = (int*)malloc(sizeof(int) * N);

    // Allocate memory on the device
    CudaSafeCall(cudaMalloc(&dA, sizeof(int) * N));
    CudaSafeCall(cudaMalloc(&dB, sizeof(int) * N));

    // Perform H2D copy
    CudaSafeCall(cudaMemcpy(dA, A, sizeof(int) * N, cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(dB, B, sizeof(int) * N, cudaMemcpyHostToDevice));

    // Launch the kernels
    coalesced<<<N/1024,1024>>>(dA, dB, N);
    stride<<<N/1024,1024>>>(dA, dB, N);

    // Clenup
    CudaSafeCall(cudaFree(dA));
    CudaSafeCall(cudaFree(dB)); 
    free(A);
    free(B);
    
    return 0;
}    
