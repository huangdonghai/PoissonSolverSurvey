
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <stdio.h>
#include <algorithm>

__global__ void JacobiIterate(float *dst, float *src, const float *b, int width, int height, int channels)
{
#define IDX(x,y,ch) ((((y)*width)+(x))*channels+ch)
#define VALID(x,y,ch) ((x)>=0&&(x)<width&&(y)>=0&&(y)<height&&ch>=0&&ch<channels)

	int x = gridDim.x * blockIdx.x + threadIdx.x;
	int y = gridDim.y * blockIdx.y + threadIdx.y;

	x = blockDim.x * blockIdx.x + threadIdx.x;
	y = blockDim.y * blockIdx.y + threadIdx.y;

	for (int ch = 0; ch < channels; ch++) {
		int count = 0;
		float n = 0, s = 0, e = 0, w = 0;

		if (!VALID(x, y, ch))
			continue;

		if (VALID(x, y + 1, ch)) {
			n = src[IDX(x, y + 1, ch)];
			count++;
		}
		if (VALID(x, y - 1, ch)) {
			s = src[IDX(x, y - 1, ch)];
			count++;
		}
		if (VALID(x + 1, y, ch)) {
			e = src[IDX(x + 1, y, ch)];
			count++;
		}
		if (VALID(x - 1, y, ch)) {
			w = src[IDX(x - 1, y, ch)];
			count++;
		}
		float center = b[IDX(x, y, ch)];


		float value = (center + (n + s + e + w)) / (float)count;

		dst[IDX(x, y, ch)] = value;
	}

#undef IDX
#undef VALID
}

cudaError_t jacobiWithCuda(int numIters, int width, int height, int channels, const float *b, float *result)
{
	cudaError_t cudaStatus;
	float* dev1 = 0;
	float* dev2 = 0;
	float* devb = 0;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	int dataSize = width * height * channels * sizeof(float);

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev1, dataSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev2, dataSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&devb, dataSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(devb, b, dataSize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	float* src = dev1;
	float* dst = dev2;
	// clear src
	cudaStatus = cudaMemset(src, 0, dataSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		goto Error;
	}

	dim3 block_dim(32, 32, 1);
	dim3 grid_dim(width, height, 1);
	grid_dim.x = (grid_dim.x + block_dim.x - 1) / block_dim.x;
	grid_dim.y = (grid_dim.y + block_dim.y - 1) / block_dim.y;
	for (int it = 0; it < numIters; it++) {

		JacobiIterate <<<grid_dim, block_dim >>> (dst, src, devb, width, height, channels);
		std::swap(src, dst);
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(result, src, dataSize, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev1);
	cudaFree(dev2);
	cudaFree(devb);

	return cudaStatus;
}
