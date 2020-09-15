//
// Created by wjx on 2020/9/10.
//

#include "internal.h"
#include "kmeans.h"

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

void benchmarkClosestPoint();

void benchmarkClosestPointInB();

void benchmarkNewCenter();

void benchmarkInitCenter();

void benchmarkKMeans();

float **twoDimensionRandom(int m, int n, float max) {
    float *buf = malloc(sizeof(float) * m * n);
    float **b = malloc(m * sizeof(float *));
    for (int i = 0; i < m; i++) {
        b[i] = buf + i * n;
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            b[i][j] = (float) rand() / RAND_MAX * max;
        }
    }
    return b;
}

void freeTwoDimension(float **arr, int l) {
    for (int i = 0; i < l; i++) {
        free(arr[i]);
    }
    free(arr);
}

int main(int argv, char *argc[]) {
    benchmarkKMeans();
    return 0;
}

void benchmarkClosestPointInB() {
    const int TestRound = 10;
    const int ALen = 100000;
    const int BLen = 400;
    const int Dimension = 21;
    const float Max = 100;

    float **a = twoDimensionRandom(ALen, Dimension, Max);
    float **b = twoDimensionRandom(BLen, Dimension, Max);

    int *result = malloc(ALen * sizeof(int));
    struct timeval now;
    gettimeofday(&now, NULL);
    for (int t = 0; t < TestRound; t++) {
        closestPointInB((const float **) a, ALen, (const float **) b, BLen, Dimension, result);
    }
    struct timeval after;
    gettimeofday(&after, NULL);
    long us = after.tv_usec < now.tv_sec ? (after.tv_sec + 1 - now.tv_sec) * 1000000 + (after.tv_usec - now.tv_usec)
                                         : (after.tv_sec - now.tv_sec) * 1000000 + (after.tv_usec - now.tv_usec);
    printf("%ld us\n", us);

    freeTwoDimension(a, ALen);
    freeTwoDimension(b, BLen);
}

void benchmarkClosestPoint() {
    const int TestRound = 2000;
    const int Size = 20000;
    const int Dimension = 21;
    const float Max = 100;

    float **arr = twoDimensionRandom(Size, Dimension, Max);

    struct timeval now;
    gettimeofday(&now, NULL);
    float dist;
    for (int t = 0; t < TestRound; t++) {
        closestPoint(arr[0], (const float **) arr, Size, Dimension, &dist);
    }
    struct timeval after;
    gettimeofday(&after, NULL);
    long us = after.tv_usec < now.tv_sec ? (after.tv_sec + 1 - now.tv_sec) * 1000000 + (after.tv_usec - now.tv_usec)
                                         : (after.tv_sec - now.tv_sec) * 1000000 + (after.tv_usec - now.tv_usec);
    printf("%ld us\n", us);
    freeTwoDimension(arr, Size);
}

void benchmarkNewCenter() {
    const int TestRound = 1000;
    const int Size = 20000;
    const int Dimension = 200;
    const int K = 30;
    const float Max = 100;
    float **p = twoDimensionRandom(Size, Dimension, Max);
    int *class = malloc(Size * sizeof(int));
    for (int i = 0; i < Size; i++) {
        class[i] = (int) ((float) rand() / RAND_MAX * K);
    }
    float **result = malloc(K * sizeof(float *));
    for (int i = 0; i < K; i++) {
        result[i] = malloc(Dimension * sizeof(float));
    }

    struct timeval now;
    gettimeofday(&now, NULL);
    for (int t = 0; t < TestRound; t++) {
        newCenter((const float **) p, Size, Dimension, class, K, result);
    }
    struct timeval after;
    gettimeofday(&after, NULL);
    long us = after.tv_usec < now.tv_sec ? (after.tv_sec + 1 - now.tv_sec) * 1000000 + (after.tv_usec - now.tv_usec)
                                         : (after.tv_sec - now.tv_sec) * 1000000 + (after.tv_usec - now.tv_usec);
    printf("%ld us\n", us);

    freeTwoDimension(p, Size);
    freeTwoDimension(result, K);
    free(class);

}

void benchmarkInitCenter() {
    const int Size = 20000;
    const int Dimension = 300;
    const int K = 30;
    const int Round = 1;

    float **data = twoDimensionRandom(Size, Dimension, 100);
    float **center = twoDimensionRandom(K, Dimension, 0);
    struct timeval now;
    gettimeofday(&now, NULL);
    for (int t = 0; t < Round; t++) {
        initCenters((const float **) data, Size, Dimension, K, (const float **) center);
    }
    struct timeval after;
    gettimeofday(&after, NULL);
    long us = after.tv_usec < now.tv_sec ? (after.tv_sec + 1 - now.tv_sec) * 1000000 + (after.tv_usec - now.tv_usec)
                                         : (after.tv_sec - now.tv_sec) * 1000000 + (after.tv_usec - now.tv_usec);
    printf("%.2f us\n", (double) us / Round);
}

void benchmarkKMeans() {
    const int Size = 50000;
    const int K = 30;
    const int Dimension = 200;

    float **data = twoDimensionRandom(Size, Dimension, 100);
    int *clazz = malloc(Size * sizeof(int));

    struct timeval now;
    gettimeofday(&now, NULL);
    kMeansConcurrent(data, Size, K, 30, Dimension, clazz);
    struct timeval after;
    gettimeofday(&after, NULL);
    long us = after.tv_usec < now.tv_sec ? (after.tv_sec + 1 - now.tv_sec) * 1000000 + (after.tv_usec - now.tv_usec)
                                         : (after.tv_sec - now.tv_sec) * 1000000 + (after.tv_usec - now.tv_usec);
    printf("%ld us\n", us);
}
