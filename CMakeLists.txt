cmake_minimum_required(VERSION 3.17)
project(kmeans_simd C)

set(CMAKE_C_STANDARD 99)

add_library(thread_pool lib/C-Thread-Pool/thpool.c)
target_link_libraries(thread_pool pthread)

add_library(internal internal.c utils.c)
target_include_directories(internal PUBLIC lib/C-Thread-Pool)
target_compile_definitions(internal PUBLIC)
target_compile_options(internal PRIVATE -mavx -mavx2 -msse -mfma)
target_link_libraries(internal thread_pool)

add_library(kmeans kmeans.c)
target_link_libraries(kmeans internal thread_pool)

# Tests and Benchmarks

add_library(internal_baseline baseline.c utils.c)
target_compile_options(internal_baseline PRIVATE -fno-tree-vectorize -fno-tree-slp-vectorize)

add_library(kmeans_baseline kmeans.c)
target_link_libraries(kmeans_baseline internal_baseline thread_pool)

add_executable(test test.c)
target_link_libraries(test kmeans)

add_executable(test_baseline test.c)
target_link_libraries(test_baseline kmeans_baseline)

add_executable(benchmark benchmark.c)
target_link_libraries(benchmark kmeans)

add_executable(benchmark_baseline benchmark.c)
target_link_libraries(benchmark_baseline kmeans_baseline)