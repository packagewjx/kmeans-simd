//
// Created by wjx on 2020/9/10.
//

#ifndef KMEANS_SIMD_SIMD_OPERATION_HPP


#if defined(__AVX512__)
#include <immintrin.h>

/**
 * Original Intrinsics
 */
#define VECTOR __m512
#define VECTOR_INT __m512i
#define ZERO_OPERATION _mm512_setzero_ps()
#define LOAD_PS_OPERATION(a) _mm512_loadu_ps(a)
#define STORE_OPERATION(addr, a) _mm512_storeu_ps(addr, a)
#define STORE_INT_OPERATION(addr, a) _mm512_storeu_epi32(addr, a)
#define NUM_CONCURRENT 16
#define ADD_OPERATION(a, b) _mm512_add_ps(a, b)
#define ADD_INT_OPERATION(a, b) _mm512_add_epi32(a, b)
#define SUBTRACT_OPERATION(a, b) _mm512_sub_ps(a, b)
#define MULTIPLY_OPERATION(a, b) _mm512_mul_ps(a, b)
#define DIV_OPERATION(a, b) _mm512_div_ps(a, b)
#define FMA_ADD_OPERATION(a, b, c) _mm512_fmadd_ps(a, b, c)
#define SET1_OPERATION(num) _mm512_set1_ps(num)
#define SET1_INT_OPERATION(num) _mm512_set1_epi32(num)
#define CMP_LT_OPERATION(a, b) _mm512_cmp_ps_mask(a, b, _CMP_LT_OS)
#define CAST_FLOAT_INT_OPERATION(a) (a)
#define MASK_DW __mmask16
#define BLEND_OPERATION(a, b, mask) _mm512_mask_blend_ps(mask, a, b)
#define BLEND_INT_OPERATION(a, b, mask) _mm512_mask_blend_epi32(mask, a, b)

/**
 * Custom Macro
 */
#define TRANSPOSE_AND_LOAD(matrix, i, j) _mm512_set_ps(                 \
matrix[i + 15][j], matrix[i + 14][j], matrix[i + 13][j], matrix[i + 12][j], \
matrix[i + 11][j], matrix[i + 10][j], matrix[i + 9][j],  matrix[i + 8][j],  \
matrix[i + 7][j],  matrix[i + 6][j],  matrix[i + 5][j],  matrix[i + 4][j],  \
matrix[i + 3][j],  matrix[i + 2][j],  matrix[i + 1][j],  matrix[i][j])
#define ZERO_INCREASE_VECTOR _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)

#elif defined(__AVX__1)
/**
 * AVX 1
 */

#include <immintrin.h>

/*
 * Original Intrinsics
 */
#define VECTOR __m256
#define VECTOR_INT __m256i
#define ZERO_OPERATION _mm256_setzero_ps()
#define LOAD_PS_OPERATION(a) _mm256_loadu_ps(a)
#define SET1_OPERATION(num) _mm256_set1_ps(num)
#define SET1_INT_OPERATION(num) _mm256_set1_epi32(num)
#define STORE_OPERATION(addr, a) _mm256_storeu_ps(addr, a)
#define STORE_INT_OPERATION(addr, a) _mm256_storeu_si256(addr, a)
#define NUM_CONCURRENT 8
#define ADD_OPERATION(a, b) _mm256_add_ps(a, b)
#define ADD_INT_OPERATION(a, b) _mm256_add_epi32(a, b)
#define SUBTRACT_OPERATION(a, b) _mm256_sub_ps(a, b)
#define MULTIPLY_OPERATION(a, b) _mm256_mul_ps(a, b)
#define DIV_OPERATION(a, b) _mm256_div_ps(a, b)
#define FMA_ADD_OPERATION(a, b, c) _mm256_fmadd_ps(a, b, c)
#define CMP_LT_OPERATION(a, b) _mm256_cmp_ps(a, b, _CMP_LT_OQ)
#define BLEND_OPERATION(a, b, mask) _mm256_blendv_ps(a, b, mask)
#define BLEND_INT_OPERATION(a, b, mask) _mm256_blendv_epi8(a, b, mask)
#define CAST_FLOAT_INT_OPERATION(a) _mm256_castps_si256(a)
#define MASK_DW __m256

/*
 * Custom Macro
 */
#define TRANSPOSE_AND_LOAD(matrix, i, j) _mm256_set_ps(matrix[i + 7][j], matrix[i + 6][j], matrix[i + 5][j], matrix[i + 4][j], \
matrix[i + 3][j], matrix[i + 2][j], matrix[i + 1][j], matrix[i][j])
#define ZERO_INCREASE_VECTOR _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0)

#elif defined(__SSE4_1__)
/**
 * SSE 1
 */
#include <immintrin.h>

/*
 * Original Intrinsics
 */
#define VECTOR __m128
#define VECTOR_INT __m128i
#define ZERO_OPERATION _mm_setzero_ps()
#define LOAD_PS_OPERATION(a) _mm_loadu_ps(a)
#define SET1_OPERATION(num) _mm_set1_ps(num)
#define SET1_INT_OPERATION(num) _mm_set1_epi32(num)
#define STORE_OPERATION(addr, a) _mm_storeu_ps(addr, a)
#define STORE_INT_OPERATION(addr, a) _mm_storeu_si128(addr, a)
#define NUM_CONCURRENT 4
#define ADD_OPERATION(a, b) _mm_add_ps(a, b)
#define ADD_INT_OPERATION(a, b) _mm_add_epi32(a, b)
#define SUBTRACT_OPERATION(a, b) _mm_sub_ps(a, b)
#define MULTIPLY_OPERATION(a, b) _mm_mul_ps(a, b)
#define DIV_OPERATION(a, b) _mm_div_ps(a, b)
#define FMA_ADD_OPERATION(a, b, c) _mm_fmadd_ps(a, b, c)
#define CMP_LT_OPERATION(a, b) _mm_cmplt_ps(a, b)
#define BLEND_OPERATION(a, b, mask) _mm_blendv_ps(a, b, mask)
#define BLEND_INT_OPERATION(a, b, mask) _mm_blendv_epi8(a, b, mask)
#define CAST_FLOAT_INT_OPERATION(a)  _mm_castps_si128(a)
#define MASK_DW __m128

/*
 * Custom Macro
 */
#define TRANSPOSE_AND_LOAD(matrix, i, j) _mm_set_ps(matrix[i + 3][j], matrix[i + 2][j], matrix[i + 1][j], matrix[i][j])
#define ZERO_INCREASE_VECTOR _mm_set_epi32(3, 2, 1, 0)

#endif

#define KMEANS_SIMD_SIMD_OPERATION_HPP

#endif //KMEANS_SIMD_SIMD_OPERATION_HPP
