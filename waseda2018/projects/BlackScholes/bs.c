#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

long long getCurrentTime()
{
    struct timeval te;
    gettimeofday(&te, NULL); // get current time
    long long microseconds = te.tv_sec*1000000LL + te.tv_usec;
    return microseconds;
}

#define N 65536

static float CND(float d)
{
    const float       A1 = 0.31938153;
    const float       A2 = -0.356563782;
    const float       A3 = 1.781477937;
    const float       A4 = -1.821255978;
    const float       A5 = 1.330274429;
    const float RSQRT2PI = 0.39894228040143267793994605993438;
    
    float K = 1.0 / (1.0 + 0.2316419 * fabsf(d));
    
   float cnd = RSQRT2PI * expf(- 0.5 * d * d) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));
    
    if (d > 0)
	cnd = 1.0 - cnd;
    
    return cnd;
}

float RandFloat(float low, float high)
{
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

int main(int argc, char **argv)
{
    int i;
    float call[N];
    float put[N];
    float price[N];
    float optS[N];
    float optY[N];

    // Initialization
    for (i = 0; i < N; i++) {
	call[i] = 0.0f;
	put[i]  = -1.0f;
	price[i] = RandFloat(5.0f, 30.0f);
        optS[i] = RandFloat(1.0f, 100.0f);
	optY[i] = RandFloat(0.25f, 10.0f);
    }

    // Computation
    for (int k = 0; k < 5; k++) {
      long long start = getCurrentTime();
      for (int i = 0; i < N; i++) {
	float S = price[i], X = optS[i], T = optY[i], R = 0.02f, V = 0.30f;
	
	float sqrtT = sqrtf(T);
	float    d1 = (logf(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT);
	float    d2 = d1 - V * sqrtT;
	float CNDD1 = CND(d1);
	float CNDD2 = CND(d2);
	
	//Calculate Call and Put simultaneously
	float expRT = expf(- R * T);
	call[i]   = (float)(S * CNDD1 - X * expRT * CNDD2);
	put[i]    = (float)(X * expRT * (1.0 - CNDD2) - S * (1.0 - CNDD1));
      }    
      long long end = getCurrentTime();
      printf("Elapsed time: %lld usec\n", (end-start));
    }
    // Print results
    printf("Call : ");
    for (i = 0; i < 10; i++) {
	printf("%f, ", call[i]);
    }
    printf("\nPut : ");
    for (i = 0; i < 10; i++) {
	printf("%f, ", put[i]);
    }
    printf("\n");	

    
}    
