#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

long long getCurrentTime() {
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

#define NDOCS 10
#define NITERS 1
#define NDIMS 2

int main(int argc, char **argv)
{
    int nLinks = 0;
    float ranks[NDOCS];
    int link_counts[NDOCS];
    int *links;
    float *link_weights;

    // Initialization
    for (int d = 0; d < NDOCS; d++) {
	ranks[d] = RandFloat(0.0f, 100.0f);
	link_counts[d] = RandFloat(5.0f, 15.0f);
	nLinks += link_counts[d];
    }
    links = (int*)malloc(sizeof(int) * NDIMS * nLinks);
    link_weights = (float*)malloc(sizeof(float) * nLinks);
    
    int current_src_doc = 0;
    int used_links = 0;
    for (int l = 0; l < nLinks; l++) {
	if (used_links == link_counts[current_src_doc]) {
	    current_src_doc++;
	    used_links = 0;
	}
	// SRC
	links[l*NDIMS+0] = current_src_doc;
	// DST
	links[l*NDIMS+1] = RandFloat(0.0f, (float)NDOCS);
	used_links++;
    }

    // Computation
    long long start = getCurrentTime();
    for (int iter = 0; iter < NITERS; iter++) {
	for (int l = 0; l < nLinks; l++) {
	    link_weights[l] = ranks[links[l*NDIMS+0]] / (float) link_counts[links[l*NDIMS+0]];
	}
	for (int d = 0; d < NDOCS; d++) {
	    float new_rank = 0.0f;
	    // look for links pointing to this document
	    for (int l = 0; l < nLinks; l++) {
		int dst = links[l*NDIMS+1];
		if (dst == d) {
		    new_rank += link_weights[l];
		}
	    }
	    ranks[d] = new_rank;
	}
    }
    long long end = getCurrentTime();

    // Print results
    printf("nDocs : %d\n", NDOCS);
    printf("nLinks : %d\n", nLinks);
    for (int d = 0; d < NDOCS; d++) {
	printf("ranks[%d] = %lf\n", d, ranks[d]);
    }
    printf("Elapsed time: %lld usec\n", (end-start));
    
}    
