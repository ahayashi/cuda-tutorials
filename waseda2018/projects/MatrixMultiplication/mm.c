#include <stdio.h>
#include <sys/time.h>

long long getCurrentTime() {
    struct timeval te;
    gettimeofday(&te, NULL); // get current time
    long long microseconds = te.tv_sec*1000000LL + te.tv_usec;
    return microseconds;
}

#define N 4

int main(int argc, char **argv)
{
    int i, j, k;
    int A[N][N];
    int B[N][N];
    int C[N][N];

    // Initialization
    for (i = 0; i < N; i++) {
	for (j = 0; j < N; j++) {
	    A[i][j] = 0;
	    B[i][j] = 1;
	    C[i][j] = 2;
	}
    }

    // Computation
    long long start = getCurrentTime();
    for (i = 0; i < N; i++) {
	for (j = 0; j < N; j++) {
	    for (k = 0; k < N; k++) {
		A[i][j] += B[i][k] * C[k][j];
	    }
	}
    }
    long long end = getCurrentTime();

    // Print results
    for (i = 0; i < N; i++) {
	for (j = 0; j < N; j++) {
	    printf("%d, ", A[i][j]);
	}
	printf("\n");
    }

    printf("Elapsed time: %lld usec\n", (end-start));
    
}    
