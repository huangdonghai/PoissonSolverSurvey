#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <algorithm>


inline void CudaCheckError()
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("ERROR_CUDA: %s\n", cudaGetErrorString(err));
	}
}

inline __device__ float3 operator+(const float3 & a, const float3 & b) {

	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);

}
inline __device__ float3 operator+=(float3 & a, const float3 & b) {

	a.x += b.x; a.y += b.y; a.z += b.z;
	return a;
}
inline __device__ float3 operator-(const float3 & a, const float3 & b) {

	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);

}
inline __device__ float3 operator*(const float3 & a, const float3 & b) {

	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);

}
inline __device__ float3 operator/(const float3 & a, const float3 & b) {

	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);

}
inline __device__ float3 operator*(const float3& a, float b) {

	return make_float3(a.x * b, a.y * b, a.z * b);

}
inline __device__ float3 operator/(const float3& a, float b) {

	return make_float3(a.x / b, a.y / b, a.z / b);

}
inline __device__ void AtomicAdd(float3 & a, const float3 & b)
{
	atomicAdd(&a.x, b.x);
	atomicAdd(&a.y, b.y);
	atomicAdd(&a.z, b.z);
}
