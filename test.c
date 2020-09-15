//
// Created by wjx on 2020/9/10.
//

#include "internal.h"
#include "stdio.h"
#include "stdlib.h"
#include "kmeans.h"

void testClosestPoint();

void testClosestPointInB();

void testNewCenter();

void testInitCenters();

void testKMeans();

float **twoDimensionRandom(int m, int n, float max) {
    float **b = malloc(sizeof(float *) * m);
    for (int i = 0; i < m; i++) {
        b[i] = malloc(sizeof(float) * n);
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
    testKMeans();
    return 0;
}

void testClosestPoint() {
    float b[25][4] = {
            {1.1f, 2.2f, 3.3f, 4.4f},
            {6,    7,    8,    9},
            {6,    7,    8,    9},
            {6,    7,    8,    9},
            {6,    7,    8,    9},
            {1,    2,    3,    4},
            {6,    7,    8,    9},
            {6,    7,    8,    9},
            {6,    7,    8,    9},
            {6,    7,    8,    9},
            {6,    7,    8,    9},
            {6,    7,    8,    9},
            {6,    7,    8,    9},
            {6,    7,    8,    9},
            {6,    7,    8,    9},
            {6,    7,    8,    9},
            {6,    7,    8,    9},
            {6,    7,    8,    9},
            {6,    7,    8,    9},
            {6,    7,    8,    9},
            {6,    7,    8,    9},
            {6,    7,    8,    9},
            {6,    7,    8,    9},
            {6,    7,    8,    9},
            {6,    7,    8,    9},
    };
    float *c[25];
    for (int i = 0; i < 25; i++) {
        c[i] = b[i];
    }

    float temp;
    printf("%d\n", closestPoint(c[0], (const float **) c, 25, 4, &temp));

}

void testClosestPointInB() {
    const int ALen = 10000;
    const int BLen = 18;
    const int Dimension = 21;
    const float Max = 100;

    float **a = malloc(sizeof(float *) * ALen);
    for (int i = 0; i < ALen; i++) {
        a[i] = malloc(sizeof(float) * Dimension);
        for (int j = 0; j < Dimension; j++) {
            a[i][j] = (float) rand() / RAND_MAX * Max;
        }
    }

    float **b = malloc(sizeof(float *) * BLen);
    for (int i = 0; i < BLen; i++) {
        b[i] = malloc(sizeof(float) * Dimension);
        for (int j = 0; j < Dimension; j++) {
            b[i][j] = (float) rand() / RAND_MAX * Max;
        }
    }

    int *result = malloc(sizeof(int) * ALen);
    closestPointInB((const float **) a, ALen, (const float **) b, BLen, Dimension, result);
    for (int i = 0; i < ALen; i++) {
        printf("%d:%d\n", i, result[i]);
    }

    for (int i = 0; i < ALen; i++) {
        free(a[i]);
    }
    free(a);
    for (int i = 0; i < BLen; i++) {
        free(b[i]);
    }
    free(b);
    free(result);
}

void testNewCenter() {
    float arr[5][9] = {
            {1, 1, 1, 1, 1, 1, 1, 1, 1},
            {2, 2, 2, 2, 2, 2, 2, 2, 2},
            {3, 3, 3, 3, 3, 3, 3, 3, 3},
            {4, 4, 4, 4, 4, 4, 4, 4, 4},
            {5, 5, 5, 5, 5, 5, 5, 5, 5},
    };
    const float *p[5] = {arr[0], arr[1], arr[2], arr[3], arr[4]};
    int clz[5] = {0, 0, 0, 1, 1};
    float result[2][9];
    float *r[2] = {result[0], result[1]};

    newCenter(p, 5, 9, clz, 2, r);

    for (int i = 0; i < 9; i++) {
        if (result[0][i] != 2) {
            printf("Wrong");
        }
        if (result[1][i] != 4.5) {
            printf("Wrong");
        }
    }

}

void testInitCenters() {
    float arr[10][9] = {
            {1, 1, 1, 1, 1, 1, 1, 1, 1},
            {2, 2, 2, 2, 2, 2, 2, 2, 2},
            {3, 3, 3, 3, 3, 3, 3, 3, 3},
            {4, 4, 4, 4, 4, 4, 4, 4, 4},
            {5, 5, 5, 5, 5, 5, 5, 5, 5},
            {6, 6, 6, 6, 6, 6, 6, 6, 6},
            {7, 7, 7, 7, 7, 7, 7, 7, 7},
            {8, 8, 8, 8, 8, 8, 8, 8, 8},
            {9, 9, 9, 9, 9, 9, 9, 9, 9},
            {10, 10, 10, 10, 10, 10, 10, 10, 10},
    };
    const float *p[10];
    for (int i = 0; i < 10; i++) {
        p[i] = arr[i];
    }
    float result[3][9];
    float *r[3] = {result[0], result[1], result[2]};

    initCenters(p, 10, 9, 3, (const float **) r);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 9; j++) {
            printf("%f ", r[i][j]);
        }
        printf("\n");
    }
}

void testKMeans() {
    const int Size = 100;
    const int Dimension = 10;
    const int K = 5;

    float **data = twoDimensionRandom(Size, Dimension, 100);
    int *clazz = malloc(Size * sizeof(int));

    kMeansConcurrent((const float **) data, Size, K, 30, Dimension, clazz);

    for (int i = 0; i < Size; i++) {
        printf("%d:%d\n", i, clazz[i]);
    }

    freeTwoDimension(data, Size);
    free(clazz);

}



