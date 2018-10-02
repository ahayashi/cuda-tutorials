#include <stdio.h>
#include <sys/time.h>

long long getCurrentTime() {
    struct timeval te;
    gettimeofday(&te, NULL); // get current time
    long long microseconds = te.tv_sec*1000000LL + te.tv_usec;
    return microseconds;
}

#define N 1024

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
    for (int iter = 0; iter < 5; iter++) {
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                A[i][j] = 0;
            }
        }
        long long start = getCurrentTime();
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                for (k = 0; k < N; k++) {
                    A[i][j] += B[i][k] * C[k][j];
                }
            }
        }
        long long end = getCurrentTime();
        printf("Elapsed time: %lld usec\n", (end-start));
    }

    // Print results
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++) {
            fprintf(stderr, "%d, ", A[i][j]);
        }
        fprintf(stderr,"\n");
    }
}
