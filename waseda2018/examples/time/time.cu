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

__global__ void assign(int *A, int *B)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    A[i] = B[i];
}

int
main()
{
    int *A, *B;
    int *dA, *dB;
    int N = 256 * 1024 * 1024;
    long long startCudaMalloc, endCudaMalloc;
    long long startCudaMemcpyH2D, endCudaMemcpyH2D;
    long long startCudaKernel, endCudaKernel;
    long long startCudaMemcpyD2H, endCudaMemcpyD2H;
    cudaEvent_t startCudaMallocEvent, endCudaMallocEvent;
    cudaEvent_t startCudaMemcpyH2DEvent, endCudaMemcpyH2DEvent;
    cudaEvent_t startCudaKernelEvent, endCudaKernelEvent;
    cudaEvent_t startCudaMemcpyD2HEvent, endCudaMemcpyD2HEvent;
    float msecTmp;
    
    CudaSafeCall(cudaEventCreate(&startCudaMallocEvent));
    CudaSafeCall(cudaEventCreate(&endCudaMallocEvent));
    CudaSafeCall(cudaEventCreate(&startCudaMemcpyH2DEvent));
    CudaSafeCall(cudaEventCreate(&endCudaMemcpyH2DEvent));
    CudaSafeCall(cudaEventCreate(&startCudaKernelEvent));
    CudaSafeCall(cudaEventCreate(&endCudaKernelEvent));
    CudaSafeCall(cudaEventCreate(&startCudaMemcpyD2HEvent));
    CudaSafeCall(cudaEventCreate(&endCudaMemcpyD2HEvent));
            
    // Step 1: Allocate memory on the host (use malloc)
    A = (int*)malloc(sizeof(int) * N);
    B = (int*)malloc(sizeof(int) * N);

    for (int i = 0; i < N; i++) {
	B[i] = i;
    }

    // Step 2: Allocate memory on the device (use cudaMalloc)
    startCudaMalloc = getCurrentTime();
    {
	CudaSafeCall(cudaEventRecord(startCudaMallocEvent));
	CudaSafeCall(cudaMalloc(&dA, sizeof(int) * N));
	CudaSafeCall(cudaMalloc(&dB, sizeof(int) * N));
	CudaSafeCall(cudaEventRecord(endCudaMallocEvent));
	CudaSafeCall(cudaEventSynchronize(endCudaMallocEvent));
    }
    endCudaMalloc = getCurrentTime();
    CudaSafeCall(cudaEventElapsedTime(&msecTmp, startCudaMallocEvent, endCudaMallocEvent));
    printf("cudaMalloc, getCurrentTime = %lf msec, cudaEventElapsedTime = %lf msec\n", (float)(endCudaMalloc-startCudaMalloc)/1000, msecTmp);
    
    // Step 3: Copy the host data to the device (use cudaMemcpy) 
    startCudaMemcpyH2D = getCurrentTime();
    {
	CudaSafeCall(cudaEventRecord(startCudaMemcpyH2DEvent));
	CudaSafeCall(cudaMemcpy(dA, A, sizeof(int) * N, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(dB, B, sizeof(int) * N, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaEventRecord(endCudaMemcpyH2DEvent));
	CudaSafeCall(cudaEventSynchronize(endCudaMemcpyH2DEvent));
    }
    endCudaMemcpyH2D = getCurrentTime();
    CudaSafeCall(cudaEventElapsedTime(&msecTmp, startCudaMemcpyH2DEvent, endCudaMemcpyH2DEvent));
    printf("cudaMemcpy, getCurrentTime = %lf msec, cudaEventElapsedTime = %lf msec\n", (float)(endCudaMemcpyH2D-startCudaMemcpyH2D)/1000, msecTmp);
    
    // Step 4: Launch the kernel
    startCudaKernel = getCurrentTime();
    {
	CudaSafeCall(cudaEventRecord(startCudaKernelEvent));
	assign<<<N/1024,1024>>>(dA, dB);
	CudaSafeCall(cudaEventRecord(endCudaKernelEvent));
	CudaSafeCall(cudaEventSynchronize(endCudaKernelEvent));
	CudaSafeCall(cudaDeviceSynchronize());
    }
    endCudaKernel = getCurrentTime();
    CudaSafeCall(cudaEventElapsedTime(&msecTmp, startCudaKernelEvent, endCudaKernelEvent));
    printf("launch, getCurrentTime = %lf msec, cudaEventElapsedTime = %lf msec\n", (float)(endCudaKernel-startCudaKernel)/1000, msecTmp);

    // Step 5: Copy back the data from the device (use cudaMemcpy)
    startCudaMemcpyD2H = getCurrentTime();
    {
	CudaSafeCall(cudaEventRecord(startCudaMemcpyD2HEvent));
	CudaSafeCall(cudaMemcpy(A, dA, sizeof(int) * N, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaEventRecord(endCudaMemcpyD2HEvent));
	CudaSafeCall(cudaEventSynchronize(endCudaMemcpyD2HEvent));
    }
    endCudaMemcpyD2H = getCurrentTime();
    CudaSafeCall(cudaEventElapsedTime(&msecTmp, startCudaMemcpyD2HEvent, endCudaMemcpyD2HEvent));
    printf("cudaMemcpy, getCurrentTime = %lf msec, cudaEventElapsedTime = %lf msec\n", (float)(endCudaMemcpyD2H-startCudaMemcpyD2H)/1000, msecTmp);
    
    // Step 6: Verification
    int error = 0;
    for (int i = 0; i < N; i++) {
	if (A[i] != i) {
	    error++;
	}
    }
    if (!error) {
	printf("VERIFIED\n");
    }
    
    // Step 7: Cleanup
    cudaFree(dA);
    cudaFree(dB);    
    free(A);
    free(B);
    
    return 0;
}    
