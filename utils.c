//
// Created by wjx on 2020/9/14.
//

#include "utils.h"

int binarySearchFloat(const float *arr, const int len, const float target) {
    if (arr[len - 1] < target) {
        return len;
    }

    int l = 0;
    int r = len - 1;
    while (l < r) {
        int m = (l + r) / 2;
        if (arr[m] < target) {
            l = m + 1;
        } else /* >= */{
            r = m;
        }
    }

    return l;
}
