
#include "Common.cuh"

__global__ void JacobiIterate(float3 *dst, float3 *src, const float3 *b, int width, int height, float omega, int it)
{
#define IDX(x,y) ((((y)*width)+(x)))
#define VALID(x,y) ((x)>=0&&(x)<width&&(y)>=0&&(y)<height)

	int x = gridDim.x * blockIdx.x + threadIdx.x;
	int y = gridDim.y * blockIdx.y + threadIdx.y;

	x = blockDim.x * blockIdx.x + threadIdx.x;
	y = blockDim.y * blockIdx.y + threadIdx.y;

	// checkerboard
	if (!VALID(x, y))
		return;

	if ((x % 2 + y) % 2 != it % 2)
		return;

	int count = 0;
	float3 n = make_float3(0,0,0), s = make_float3(0, 0, 0), e = make_float3(0, 0, 0), w = make_float3(0, 0, 0);

	if (VALID(x, y + 1)) {
		n = src[IDX(x, y + 1)];
		count++;
	}
	if (VALID(x, y - 1)) {
		s = src[IDX(x, y - 1)];
		count++;
	}
	if (VALID(x + 1, y)) {
		e = src[IDX(x + 1, y)];
		count++;
	}
	if (VALID(x - 1, y)) {
		w = src[IDX(x - 1, y)];
		count++;
	}
	float3 bSelf = b[IDX(x, y)];
	float3 last = src[IDX(x, y)];

	float3 value = (bSelf + (n + s + e + w)) / (float)count;
	value = last * (1.0 - omega) + value * omega;

	dst[IDX(x, y)] = value;

#undef IDX
#undef VALID
}

// out = vector + vector
__global__ void add(const float3* a, const float3* b, float3* out, int size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < size)
		out[idx] = a[idx] + b[idx];
}

cudaError_t jacobiWithCuda(int numIters, int width, int height, int channels, const float *b, float *result, bool isSOR)
{
	int pixelCount = width * height;
	int blockDim1 = 1024;
	int gridDim1 = (pixelCount + blockDim1 - 1) / blockDim1;

	float3* dev1 = 0;
	float3* dev2 = 0;
	float3* devb = 0;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaSetDevice(0);

	int dataSize = width * height * channels * sizeof(float);

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaMalloc((void**)&dev1, dataSize);
	cudaMalloc((void**)&dev2, dataSize);
	cudaMalloc((void**)&devb, dataSize);

	// Copy input vectors from host memory to GPU buffers.
	cudaMemcpy(devb, b, dataSize, cudaMemcpyHostToDevice);

	float3* src = dev1;
	float3* dst = dev2;
	// clear src
	cudaMemset(src, 0, dataSize);
	cudaMemset(dst, 0, dataSize);

	dim3 block_dim(32, 32, 1);
	dim3 grid_dim(width, height, 1);
	grid_dim.x = (grid_dim.x + block_dim.x - 1) / block_dim.x;
	grid_dim.y = (grid_dim.y + block_dim.y - 1) / block_dim.y;

	float omega = 1;
	const float pi = 3.1415926;
	float ro = 1 - pi * pi / (4.0 * 32 * 32); //spectral radius;
	if (isSOR) {
		omega = 1.005;
	}
	for (int it = 0; it < numIters; it++) {

		JacobiIterate <<<grid_dim, block_dim>>> (dst, src, devb, width, height, omega, it);
		//omega -= 1.0f/numIters;
		std::swap(src, dst);
	}
	add<<<gridDim1, blockDim1>>>(src, dst, src, pixelCount);
	CudaCheckError();

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaDeviceSynchronize();

	cudaMemcpy(result, src, dataSize, cudaMemcpyDeviceToHost);

	cudaFree(dev1);
	cudaFree(dev2);
	cudaFree(devb);

	return cudaSuccess;
}
