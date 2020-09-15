//
// Created by wjx on 2020/9/10.
//

#ifndef KMEANS_SIMD_KMEANS_H

void kMeans(const float **data, int len, int k, int round, int d, int *clazz);

void kMeansConcurrent(const float **data, int len, int k, int round, int d, int *clazz);

#define KMEANS_SIMD_KMEANS_H

#endif //KMEANS_SIMD_KMEANS_H
