#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <algorithm>


#ifndef DISABLE_ERROR_CHECK
#define CUDA_CHECK_ERR() \
  do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
      printf("%s:%d:%s\n ERROR_CUDA: %s\n", __FILE__, __LINE__, __func__, \
             cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)
#else
#define CUDA_CHECK_ERR()
#endif
