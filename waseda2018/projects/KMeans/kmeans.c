#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

long long getCurrentTime()
{
    struct timeval te;
    gettimeofday(&te, NULL); // get current time
    long long microseconds = te.tv_sec*1000000LL + te.tv_usec;
    return microseconds;
}

float RandFloat(float low, float high)
{
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

#define NPOINTS 65536
#define NCLUSTERS 3
#define NDIMS 1
#define NITERS 5

int main(int argc, char **argv)
{
    float points[NPOINTS][NDIMS];
    float centers[NCLUSTERS][NDIMS];
    int classifications[NPOINTS];

    // Initialization
    for (int p = 0; p < NPOINTS; p++) {
	for (int d = 0; d < NDIMS; d++) {
	    points[p][d] = RandFloat(1.0f, 10.0f);
	}
    }
    for (int c = 0; c < NCLUSTERS; c++) {
	for (int d = 0; d < NDIMS; d++) {
	    centers[c][d] = points[c][d]; // assuming NCLUSTERS <= NPOINTS
	}
    }

    // Computation
    for (int iter = 0; iter < NITERS; iter++) {
	long long start = getCurrentTime();
	for (int p = 0; p < NPOINTS; p++) {
	    int bestCluster = 0;
	    float bestDistance = 0;
	    
	    for (int c = 0; c < NCLUSTERS; c++) {
		float dist = 0.0;
		for (int d = 0; d < NDIMS; d++) {
		    double diff = points[p][d] - centers[c][d];
		    dist += diff * diff;
		}
		if (c == 0 || bestDistance > dist) {
		    bestCluster  = c;
		    bestDistance = dist;
		}
	    }
	    classifications[p] = bestCluster;
	}
	for (int c = 0; c < NCLUSTERS; c++) {
	    int count = 0;
	    for (int d = 0; d < NDIMS; d++) {
		centers[c][d] = 0.0;
	    }
	    for (int p = 0; p < NPOINTS; p++) {
		if (classifications[p] == c) {
		    for (int d = 0; d < NDIMS; d++) {
			centers[c][d] += points[p][d];
		    }
		    count++;
		}
	    }
	    for (int d = 0; d < NDIMS; d++) {
		centers[c][d] = centers[c][d] / (double)count;
	    }
	}
	long long end = getCurrentTime();
	printf("Elapsed time: %lld usec\n", (end-start));
    }

    // Print results
    printf("nPoints : %d\n", NPOINTS);
    printf("nClusters : %d\n", NCLUSTERS);
    printf("nDimensions : %d\n", NDIMS);
    for (int p = 0; p < 10; p++) {
	printf("P%d : %d, ", p, classifications[p]);
    }
    printf("\n");    
}    
