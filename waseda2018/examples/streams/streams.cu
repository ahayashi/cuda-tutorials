#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>

long long getCurrentTime()
{
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
    if ( cudaSuccess != err ) {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif
    return;
}

__global__ void assign(int *A, int streamId)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < 1024 * 1024;
         i += blockDim.x * gridDim.x) {
        A[i] += 1024*1024*streamId + i;
    }
}

int main()
{
    cudaStream_t streams[4];
    int *A[4], *dA[4];
    int N = 1024*1024;

    // Initialization
    for (int i = 0; i < 4; i++) {
        CudaSafeCall(cudaStreamCreate(&streams[i]));
        CudaSafeCall(cudaMallocHost((void**)&A[i], sizeof(int) * N));
        CudaSafeCall(cudaMalloc(&dA[i], N * sizeof(int)));
    }

    // Asynchronous H2D copies
    for (int i = 0; i < 4; i++) {
        CudaSafeCall(cudaMemcpyAsync(dA[i], A[i], N * sizeof(int),
                                     cudaMemcpyHostToDevice, streams[i]));
    }

    // Asynchronous kernel launches
    for (int i = 0; i < 4; i++) {
        assign<<<1, 1024, 0, streams[i]>>>(dA[i], i);
    }

    // Asynchronous D2H copies
    for (int i = 0; i < 4; i++) {
        CudaSafeCall(cudaMemcpyAsync(A[i], dA[i], N * sizeof(int),
                                     cudaMemcpyDeviceToHost, streams[i]));
    }

    // Synchronization
    for (int i = 0; i < 4; i++) {
        CudaSafeCall(cudaStreamSynchronize(streams[i]));
    }

    // Verification
    int error = 0;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < N; j++) {
            if (A[i][j] != i * N + j) {
                error++;
            }
        }
    }
    if (!error) {
        printf("VERIFILED\n");
    } else {
        printf("NOT VERIFIED");
    }

    return 0;
}
