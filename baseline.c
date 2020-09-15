//
// Created by wjx on 2020/9/10.
//

#include "internal.h"
#include "utils.h"

#include <stdlib.h>
#include <memory.h>

int closestPoint(const float *p, const float **points, const int len, const int d, float *closestDist) {
    int closest = -1;
    *closestDist = 3.402823466e+38F;

    for (int i = 0; i < len; i++) {
        float dist = 0;
        for (int j = 0; j < d; j++) {
            float temp = p[j] - points[i][j];
            dist += temp * temp;
        }
        if (dist < *closestDist) {
            *closestDist = dist;
            closest = i;
        }
    }

    return closest;
}

void closestPointInB(const float **a, const int aLen, const float **b, const int bLen, const int d, int *result) {
    float temp;
    for (int i = 0; i < aLen; i++) {
        result[i] = closestPoint(a[i], b, bLen, d, &temp);
    }
}

void
closestPointInBConcurrent(const float **a, int aLen, const float **b, int bLen, int d, int *result, threadpool pool) {
    closestPointInB(a, aLen, b, bLen, d, result);
}


void newCenter(const float **points, const int len, const int d, const int *clazz, const int k, float **result) {
    for (int i = 0; i < k; i++) {
        memset(result[i], 0, d * sizeof(float));
    }

    int *count = calloc(len, sizeof(int));

    for (int i = 0; i < len; i++) {
        int clz = clazz[i];
        count[clz]++;
        for (int j = 0; j < d; j++) {
            result[clz][j] += points[i][j];
        }
    }

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < d; j++) {
            result[i][j] /= (float) count[i];
        }
    }
    free(count);
}

void initCenters(const float **points, int len, int d, int k, const float **centers) {
    int initialCenter = (int) ((float) rand() / RAND_MAX * len);
    centers[0] = points[initialCenter];

    float *closestDist = malloc(len * sizeof(float));
    for (int i = 1; i < k; i++) {
        float distSum = 0;
        for (int j = 0; j < len; j++) {
            closestPoint(points[j], centers, i, d, closestDist + j);
            distSum += closestDist[j];
        }

        closestDist[0] = closestDist[0] / distSum;
        for (int j = 1; j < len; j++) {
            closestDist[j] = closestDist[j] / distSum + closestDist[j - 1];
        }

        // Change last elements to 1 due to float addition error.
        for (int j = len - 2; j >= 0; j--) {
            if (closestDist[j] == closestDist[len - 1]) {
                closestDist[j] = 1;
            }
        }
        closestDist[len - 1] = 1;

        // Randomly pick a center
        float p = (float) rand() / RAND_MAX;
        int idx = binarySearchFloat(closestDist, len, p);
        centers[i] = points[idx];
    }

    free(closestDist);
}

void initCentersConcurrent(const float **points, int len, int d, int k, const float **centers, threadpool pool) {
    initCenters(points, len, d, k, centers);
}