//
// Created by wjx on 2020/9/10.
//

#include "internal.h"
#include "utils.h"
#include "simd_operation.h"

#include <stdlib.h>
#include <memory.h>
#include <sys/time.h>
#include <pthread.h>

#define FLT_MAX         3.402823466e+38F        /* float max value */

float distanceSquare(const float *p, const float *q, const int d) {
    float dist = 0;
    VECTOR addResult = ZERO_OPERATION;

    int i = 0;
    for (; i + NUM_CONCURRENT - 1 < d; i += NUM_CONCURRENT) {
        VECTOR pi = LOAD_PS_OPERATION(p + i);
        VECTOR qi = LOAD_PS_OPERATION(q + i);
        pi = SUBTRACT_OPERATION(pi, qi);
        addResult = FMA_ADD_OPERATION(pi, pi, addResult);
    }

    float *store = malloc(sizeof(float) * NUM_CONCURRENT);
    STORE_OPERATION(store, addResult);
    for (int j = 0; j < NUM_CONCURRENT; j++) {
        dist += store[j];
    }
    free(store);

    // Handle last several elements
    for (; i < d; i++) {
        float temp = p[i] - q[i];
        dist += temp * temp;
    }

    return dist;
}

int closestPoint(const float *p, const float **points, const int len, const int d, float *closestDist) {
    int i = 0;

    const VECTOR_INT Increment = SET1_INT_OPERATION(NUM_CONCURRENT);
    VECTOR minValues = SET1_OPERATION(FLT_MAX);
    VECTOR_INT indices = ZERO_INCREASE_VECTOR;
    VECTOR_INT minIndices = SET1_INT_OPERATION(-1);

    for (; i + NUM_CONCURRENT - 1 < len; i += NUM_CONCURRENT) {
        VECTOR frontHalf = ZERO_OPERATION;
        VECTOR rearHalf = ZERO_OPERATION;
        int split = d / 2;

        // FMA, Load, Subtract has a CPI of 0.5.
        for (int j = 0; j < split; j++) {
            // front half
            VECTOR a = TRANSPOSE_AND_LOAD(points, i, j);
            VECTOR b = SET1_OPERATION(p[j]);
            a = SUBTRACT_OPERATION(a, b);
            frontHalf = FMA_ADD_OPERATION(a, a, frontHalf);

            // rear half
            VECTOR x = TRANSPOSE_AND_LOAD(points, i, j + split);
            VECTOR y = SET1_OPERATION(p[j + split]);
            x = SUBTRACT_OPERATION(x, y);
            rearHalf = FMA_ADD_OPERATION(x, x, rearHalf);
        }
        // Last Dimension
        if ((d & 1) == 1) {
            VECTOR a = TRANSPOSE_AND_LOAD(points, i, d - 1);
            VECTOR b = SET1_OPERATION(p[d - 1]);
            a = SUBTRACT_OPERATION(a, b);
            frontHalf = FMA_ADD_OPERATION(a, a, frontHalf);
        }
        VECTOR addResult = ADD_OPERATION(frontHalf, rearHalf);

        // 更新最小值
        MASK_DW lessMask = CMP_LT_OPERATION(addResult, minValues);
        minValues = BLEND_OPERATION(minValues, addResult, lessMask);
        minIndices = BLEND_INT_OPERATION(minIndices, indices, CAST_FLOAT_INT_OPERATION(lessMask));
        indices = ADD_INT_OPERATION(indices, Increment);
    }

    float *vector = malloc(sizeof(float) * NUM_CONCURRENT);
    int32_t *vectorInt = malloc(sizeof(int32_t) * NUM_CONCURRENT);
    STORE_OPERATION(vector, minValues);
    STORE_INT_OPERATION((VECTOR_INT *) vectorInt, minIndices);

    // 取得最小的
    int closest = -1;
    *closestDist = FLT_MAX;
    for (int j = 0; j < NUM_CONCURRENT; j++) {
        if (vector[j] < *closestDist) {
            *closestDist = vector[j];
            closest = vectorInt[j];
        }
    }

    for (; i < len; i++) {
        float dst = distanceSquare(p, points[i], d);
        if (dst < *closestDist) {
            closest = i;
            *closestDist = dst;
        }
    }
    free(vector);
    free(vectorInt);

    return closest;
}

void closestPointInB(const float **a, const int aLen, const float **b, const int bLen, const int d, int *result) {
    float temp;
    for (int i = 0; i < aLen; i++) {
        result[i] = closestPoint(a[i], b, bLen, d, &temp);
    }
}

typedef struct {
    const float *p;
    const float **points;
    const int len;
    const int d;
    int *result;
} ClosestPointArg;

void closestPointWorker(void *pointer) {
    ClosestPointArg *arg = pointer;
    float temp;
    *arg->result = closestPoint(arg->p, arg->points, arg->len, arg->d, &temp);
    free(pointer);
}

void
closestPointInBConcurrent(const float **a, const int aLen, const float **b, const int bLen, const int d,
                          int *result, threadpool pool) {
    for (int i = 0; i < aLen; i++) {
        ClosestPointArg *arg = malloc(sizeof(ClosestPointArg));
        arg->p = a[i];
        arg->points = b;
        *(int *) &arg->len = bLen;
        *(int *) &arg->d = d;
        arg->result = result + i;
        thpool_add_work(pool, closestPointWorker, arg);
    }

    thpool_wait(pool);
}

void newCenter(const float **points, const int len, const int d, const int *clazz, const int k, float **result) {
    for (int i = 0; i < k; i++) {
        memset(result[i], 0, d * sizeof(float));
    }

    int *count = calloc(len, sizeof(int));
    for (int i = 0; i < len; i++) {
        int clz = clazz[i];
        count[clz]++;
        int j = 0;
        // Load, Add has CPI 0.5.
        for (; j + 2 * NUM_CONCURRENT - 1 < d; j += 2 * NUM_CONCURRENT) {
            VECTOR p1 = LOAD_PS_OPERATION(points[i] + j);
            VECTOR c1 = LOAD_PS_OPERATION(result[clz] + j);
            VECTOR p2 = LOAD_PS_OPERATION(points[i] + j + NUM_CONCURRENT);
            VECTOR c2 = LOAD_PS_OPERATION(result[clz] + j + NUM_CONCURRENT);
            c1 = ADD_OPERATION(p1, c1);
            c2 = ADD_OPERATION(p2, c2);
            STORE_OPERATION(result[clz] + j, c1);
            STORE_OPERATION(result[clz] + j + NUM_CONCURRENT, c2);
        }
        for (; j < d; j++) {
            result[clz][j] += points[i][j];
        }
    }

    for (int i = 0; i < k; i++) {
        int j = 0;
        VECTOR c = SET1_OPERATION(count[i]);
        for (; j + NUM_CONCURRENT - 1 < d; j += NUM_CONCURRENT) {
            VECTOR p1 = LOAD_PS_OPERATION(result[i] + j);
            p1 = DIV_OPERATION(p1, c);
            STORE_OPERATION(result[i] + j, p1);
        }
        for (; j < d; j++) {
            result[i][j] /= (float) count[i];
        }
    }

    free(count);
}

/**
 * Choose a center using K-Means++ method.
 * @param closestDist Closest distance to the closest center.
 * @param distSum sum of the closestDist.
 * @return center index.
 */
int chooseCenter(float *closestDist, int len, float distSum) {// Probability Calculation
    closestDist[0] = closestDist[0] / distSum;
    VECTOR sumVector = SET1_OPERATION(distSum);
    int j = 1;
    for (; j + NUM_CONCURRENT - 1 < len; j += NUM_CONCURRENT) {
        VECTOR dist = LOAD_PS_OPERATION(closestDist + j);
        dist = DIV_OPERATION(dist, sumVector);
        STORE_OPERATION(closestDist + j, dist);
        closestDist[j] += closestDist[j - 1];
        for (int l = 1; l < NUM_CONCURRENT; l++) {
            closestDist[j + l] += closestDist[j + l - 1];
        }
    }

    for (; j < len; j++) {
        closestDist[j] = (closestDist[j] / distSum) + closestDist[j - 1];
    }

    // Change last elements to 1 due to float addition error.
    j = len - 2;
    for (; j >= 0; j--) {
        if (closestDist[j] == closestDist[len - 1]) {
            closestDist[j] = 1;
        } else {
            break;
        }
    }
    closestDist[len - 1] = 1;

    // Randomly pick a center
    float p = (float) rand() / RAND_MAX;
    int idx = binarySearchFloat(closestDist, len, p);
    return idx;
}


void initCenters(const float **points, const int len, const int d, const int k, const float **centers) {
    int initialCenter = (int) ((float) rand() / RAND_MAX * len);
    centers[0] = points[initialCenter];

    float *closestDist = malloc(len * sizeof(float));
    for (int i = 1; i < k; i++) {
        float distSum = 0;
        for (int j = 0; j < len; j++) {
            closestPoint(points[j], centers, i, d, closestDist + j);
            distSum += closestDist[j];
        }
        int idx = chooseCenter(closestDist, len, distSum);
        centers[i] = points[idx];
    }

    free(closestDist);
}

typedef struct {
    const float *p;
    const int d;
    const float **centers;
    const int len;
    float *distSum;
    float *closestDist;
    pthread_mutex_t *lock;
} InitCenterArg;

void initCentersWorker(void *vArg) {
    InitCenterArg *arg = vArg;
    closestPoint(arg->p, arg->centers, arg->len, arg->d, arg->closestDist);

    pthread_mutex_lock(arg->lock);
    *arg->distSum += *arg->closestDist;
    pthread_mutex_unlock(arg->lock);

    free(arg);
}

void initCentersConcurrent(const float **points, int len, int d, int k, const float **centers, threadpool pool) {
    int initialCenter = (int) ((float) rand() / RAND_MAX * len);
    centers[0] = points[initialCenter];

    pthread_mutex_t lock;
    pthread_mutex_init(&lock, NULL);
    float *closestDist = malloc(len * sizeof(float));
    for (int i = 1; i < k; i++) {
        float distSum = 0;

        for (int j = 0; j < len; j++) {
            InitCenterArg *arg = malloc(sizeof(InitCenterArg));
            arg->p = points[j];
            arg->centers = centers;
            *(int *) &arg->len = i;
            *(int *) &arg->d = d;
            arg->lock = &lock;
            arg->distSum = &distSum;
            arg->closestDist = closestDist + j;

            thpool_add_work(pool, initCentersWorker, arg);
        }
        thpool_wait(pool);

        int idx = chooseCenter(closestDist, len, distSum);
        centers[i] = points[idx];
    }

    free(closestDist);
}
