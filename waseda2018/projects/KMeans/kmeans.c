#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
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

char* readline(FILE *input, char *line, int *max_line_len)
{
    int len;

    if(fgets(line,*max_line_len,input) == NULL)
        return NULL;

    while(strrchr(line,'\n') == NULL)
        {
            *max_line_len *= 2;
            line = (char *) realloc(line, *max_line_len);
            len = (int) strlen(line);
            if(fgets(line+len,*max_line_len-len,input) == NULL)
                break;
        }
    return line;
}

int main(int argc, char **argv)
{
    float *points;
    float *centers;
    int *classifications;

    int NPOINTS;
    int NDIMS;
    int NCLUSTERS = 2048;
    int NITERS = 25;

    // Initialization
    if (argc == 1) {
        printf("use random data\n");
        NPOINTS = 65536;
        NDIMS = 2;

        points = (float*)malloc(sizeof(float)*NPOINTS*NDIMS);
        for (int p = 0; p < NPOINTS; p++) {
            for (int d = 0; d < NDIMS; d++) {
                points[p*NDIMS+d] = 1 + RandFloat(0.01f, 1.05f);
            }
        }
    } else if (argc == 2) {
        printf("reading %s\n", argv[1]);
        FILE *fp = fopen(argv[1], "r");
        int max_line_len = 1024;
        char *line = (char *)malloc(max_line_len*sizeof(char));
        if (fp == NULL) {
            printf("File not found: %s\n", argv[1]);
            abort();
        }
        NPOINTS = 0;
        NDIMS = 0;
        while(readline(fp, line, &max_line_len)) {
            int idx;
            float val;
            char *p = line;
            while (isspace(*p)) ++p;
            while (!isspace(*p)) ++p;
            while (sscanf(p, "%d:%f", &idx, &val) == 2) {
                if (idx > NDIMS) {
                    NDIMS = idx;
                }
                while(*p!=':') ++p;
                ++p;
                while(isspace(*p)) ++p;
                while(*p && !isspace(*p)) ++p;
            }
            NPOINTS++;
        }
        int pidx = 0;
        points = (float*)calloc(NPOINTS*NDIMS, sizeof(float));
        fseek(fp, 0, SEEK_SET);
        while(readline(fp, line, &max_line_len)) {
            int idx;
            float val;
            char *p = line;
            while (isspace(*p)) ++p;
            while (!isspace(*p)) ++p;
            while (sscanf(p, "%d:%f", &idx, &val) == 2) {
                points[pidx*NDIMS+(idx-1)] = val;
                while(*p!=':') ++p;
                ++p;
                while(isspace(*p)) ++p;
                while(*p && !isspace(*p)) ++p;
            }
            pidx++;
        }
    }

    printf("[Summary]\n");
    printf("nPoints: %d\n", NPOINTS);
    printf("nDims: %d\n", NDIMS);
    printf("nClusters: %d\n", NCLUSTERS);
    printf("nIters: %d\n", NITERS);

#if 0
    for (int p = 0; p < 10; p++) {
        printf("P%d: {", p);
        for (int d = 0; d < NDIMS; d++) {
            printf("%d:%lf, ", d, points[p*NDIMS+d]);
        }
        printf("}\n", p);
    }
#endif

    centers = (float*)malloc(sizeof(float)*NCLUSTERS*NDIMS);
    classifications = (int*)malloc(sizeof(int)*NPOINTS);

    for (int c = 0; c < NCLUSTERS; c++) {
        for (int d = 0; d < NDIMS; d++) {
            centers[c*NDIMS+d] = points[c*NDIMS+d]; // assuming NCLUSTERS <= NPOINTS
        }
    }

    // Computation
    long long startall = getCurrentTime();
    for (int iter = 0; iter < NITERS; iter++) {
        long long start = getCurrentTime();
        for (int p = 0; p < NPOINTS; p++) {
            int bestCluster = 0;
            float bestDistance = 0;

            for (int c = 0; c < NCLUSTERS; c++) {
                float dist = 0.0;
                for (int d = 0; d < NDIMS; d++) {
                    double diff = points[p*NDIMS+d] - centers[c*NDIMS+d];
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
                centers[c*NDIMS+d] = 0.0;
            }
            for (int p = 0; p < NPOINTS; p++) {
                if (classifications[p] == c) {
                    for (int d = 0; d < NDIMS; d++) {
                        centers[c*NDIMS+d] += points[p*NDIMS+d];
                    }
                    count++;
                }
            }
            for (int d = 0; d < NDIMS; d++) {
                centers[c*NDIMS+d] = centers[c*NDIMS+d] / (double)count;
            }
        }
        long long end = getCurrentTime();
        printf("Elapsed time: %lld usec\n", (end-start));
    }
    long long endall = getCurrentTime();
    printf("Elapsed time: %lf sec\n", (endall-startall)*1.0/1000000);

    // Print results
    printf("nPoints : %d\n", NPOINTS);
    printf("nClusters : %d\n", NCLUSTERS);
    printf("nDimensions : %d\n", NDIMS);
    // only put the first 10 results
    for (int p = 0; p < 10; p++) {
        printf("P%d : %d, ", p, classifications[p]);
    }
    printf("\n");
}
