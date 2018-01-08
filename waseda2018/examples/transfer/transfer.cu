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

int
main()
{


    for (int N = 1024; N < 256 * 1024* 1024; N = N * 2) {
	int *pageA, *pageB, *pinnedA, *pinnedB;
	int *dA;

	//
	pageA = (int*)malloc(sizeof(int) * N);
	pageB = (int*)malloc(sizeof(int) * N);
	CudaSafeCall(cudaMalloc(&dA, sizeof(int) * N));

	//
	CudaSafeCall(cudaMallocHost((void**)&pinnedA, sizeof(int) * N));
	CudaSafeCall(cudaMallocHost((void**)&pinnedB, sizeof(int) * N));
	
	//
	for (int i = 0; i < N; i++) {
	    pageA[i] = i;
	    pinnedA[i] = i;
	}

	//
	cudaEvent_t startH2DPage, endH2DPage;
	cudaEvent_t startD2HPage, endD2HPage;
	CudaSafeCall(cudaEventCreate(&startH2DPage));
	CudaSafeCall(cudaEventCreate(&endH2DPage));
	CudaSafeCall(cudaEventCreate(&startD2HPage));
	CudaSafeCall(cudaEventCreate(&endD2HPage));

	CudaSafeCall(cudaEventRecord(startH2DPage));	
	CudaSafeCall(cudaMemcpy(dA, pageA, sizeof(int) * N, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaEventRecord(endH2DPage));
	CudaSafeCall(cudaEventSynchronize(endH2DPage));

	CudaSafeCall(cudaEventRecord(startD2HPage));	
	CudaSafeCall(cudaMemcpy(pageB, dA, sizeof(int) * N, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaEventRecord(endD2HPage));
	CudaSafeCall(cudaEventSynchronize(endD2HPage));
       
	int error = 0;
	for (int i = 0; i < N; i++) {
	    if (pageA[i] != pageB[i]) {
		error++;
	    }
	}

	//
	cudaEvent_t startH2DPinned, endH2DPinned;
	cudaEvent_t startD2HPinned, endD2HPinned;
	CudaSafeCall(cudaEventCreate(&startH2DPinned));
	CudaSafeCall(cudaEventCreate(&endH2DPinned));
	CudaSafeCall(cudaEventCreate(&startD2HPinned));
	CudaSafeCall(cudaEventCreate(&endD2HPinned));

	CudaSafeCall(cudaEventRecord(startH2DPinned));		
	CudaSafeCall(cudaMemcpy(dA, pinnedA, sizeof(int) * N, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaEventRecord(endH2DPinned));
	CudaSafeCall(cudaEventSynchronize(endH2DPinned));

	CudaSafeCall(cudaEventRecord(startD2HPinned));		
	CudaSafeCall(cudaMemcpy(pinnedB, dA, sizeof(int) * N, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaEventRecord(endD2HPinned));
	CudaSafeCall(cudaEventSynchronize(endD2HPinned));

	for (int i = 0; i < N; i++) {
	    if (pinnedA[i] != pinnedB[i]) {
		error++;
	    }
	}
	
	if (!error) {
	    float h2dpage, d2hpage, h2dpinned, d2hpinned;
	    CudaSafeCall(cudaEventElapsedTime(&h2dpage, startH2DPage, endH2DPage));
	    CudaSafeCall(cudaEventElapsedTime(&d2hpage, startD2HPage, endD2HPage));
	    CudaSafeCall(cudaEventElapsedTime(&h2dpinned, startH2DPinned, endH2DPinned));
	    CudaSafeCall(cudaEventElapsedTime(&d2hpinned, startD2HPinned, endD2HPinned));
	    printf("Size: %lu bytes, H2D Page: %lf msec, D2H Page: %lf msec, H2D Pinned: %lf msec, D2H Pinned: %lf msec\n", N * sizeof(int), h2dpage, d2hpage, h2dpinned, d2hpinned);
	}
    }
    
    return 0;
}    
