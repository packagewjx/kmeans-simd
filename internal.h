//
// Created by wjx on 2020/9/10.
//

#ifndef KMEANS_SIMD_INTERNAL_H

#include "lib/C-Thread-Pool/thpool.h"

/**
 * Calculate the euclidean distanceSquare between p and q both with d features(dimensions).
 */
float distanceSquare(const float *p, const float *q, int d);

/**
 * Find the closest point for p in points array, return the distance and index of the point.
 * If points contains p, it WILL return the index of p, and closestDist is 0.
 */
int closestPoint(const float *p, const float **points, int len, int d, float *closestDist);

/**
 * Find the closest point in b for every point in a, and store it in result array.
 */
void closestPointInB(const float **a, int aLen, const float **b, int bLen, int d, int *result);

/**
 * Find the closest point in b for every point in a, and store it in result array, concurrently.
 */
void
closestPointInBConcurrent(const float **a, int aLen, const float **b, int bLen, int d, int *result, threadpool pool);

/**
 * Calculate new centers based on classification result.
 * @param clazz class of every point.
 * @param result buffer to place new centers. It MUST be a 2-D array, not a float* array.
 */
void newCenter(const float **points, int len, int d, const int *clazz, int k, float **result);

/**
 * Initialize center using K-Means++ method
 * @param centers store the center
 */
void initCenters(const float **points, int len, int d, int k, const float **centers);

/**
 * Initialize center using K-Means++ method concurrently.
 * @param centers store the center
 */
void initCentersConcurrent(const float **points, int len, int d, int k, const float **centers, threadpool pool);

#define KMEANS_SIMD_INTERNAL_H

#endif //KMEANS_SIMD_INTERNAL_H
