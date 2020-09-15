//
// Created by wjx on 2020/9/14.
//

#include "kmeans.h"
#include "internal.h"
#include "lib/C-Thread-Pool/thpool.h"

#include <stdlib.h>
#include <memory.h>
#include <unistd.h>

/**
 * K-Means++
 * @param round the number of iteration round
 * @param d the number of features
 * @param clazz store the result of K-Means
 */
void kMeans(const float **data, const int len, const int k, const int round, const int d, int *clazz) {
    float **centers = malloc(k * sizeof(float *));
    initCenters(data, len, d, k, (const float **) centers);

    // We have to copy here because initCenters will set centers to points in data array, and centers will be changed
    // in iterations.
    for (int i = 0; i < k; i++) {
        float *c = malloc(d * sizeof(float));
        memcpy(c, centers[i], d * sizeof(float));
        centers[i] = c;
    }

    for (int t = 0; t < round; t++) {
        closestPointInB(data, len, (const float **) centers, k, d, clazz);
        newCenter(data, len, d, clazz, k, centers);
    }

    for (int i = 0; i < k; i++) {
        free(centers[i]);
    }
    free(centers);
}

void kMeansConcurrent(const float **data, const int len, const int k, const int round, const int d, int *clazz) {
    threadpool pool = thpool_init((int) sysconf(_SC_NPROCESSORS_ONLN));
    float **centers = malloc(k * sizeof(float *));
    initCentersConcurrent(data, len, d, k, (const float **) centers, pool);

    // We have to copy here because initCenters will set centers to points in data array, and centers will be changed
    // in iterations.
    for (int i = 0; i < k; i++) {
        float *c = malloc(d * sizeof(float));
        memcpy(c, centers[i], d * sizeof(float));
        centers[i] = c;
    }

    for (int t = 0; t < round; t++) {
        closestPointInBConcurrent(data, len, (const float **) centers, k, d, clazz, pool);
        newCenter(data, len, d, clazz, k, centers);
    }

    for (int i = 0; i < k; i++) {
        free(centers[i]);
    }
    free(centers);

}