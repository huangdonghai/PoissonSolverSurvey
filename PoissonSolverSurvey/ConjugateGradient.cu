
#include "Common.cuh"
#include <assert.h>

// Dot product of two vector
__global__ void dot(float3* a, float3* b, float3* out, int size) {
	// each block has it's own shared_tmp of dynamic size n
	extern __shared__ float3 shared_tmp[];

	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	// needed for atomicAdd
	if (idx == 0) {
		*out = make_float3(0,0,0);
	}


	if (idx < size) {
		shared_tmp[threadIdx.x] = a[idx] * b[idx];
	} else {
		// needed for the reduction
		shared_tmp[threadIdx.x] = make_float3(0,0,0);
	}

	// reduction within block
	for (int i = blockDim.x / 2; i >= 1; i = i / 2) {
		// threads access memory position written by other threads so sync is needed
		__syncthreads();
		if (threadIdx.x < i) {
			shared_tmp[threadIdx.x] += shared_tmp[threadIdx.x + i];
		}
	}

	// atomic add the partial reduction in out
	if (threadIdx.x == 0) {
		AtomicAdd(out[0], shared_tmp[0]);
	}
}

__global__ void Amulv(const float3* src, float3* Ap, int width, int height)
{
#define IDX(x,y) ((y)*width+(x))
#define VALID(x,y) ((x)>=0&&(x)<width&&(y)>=0&&(y)<height)

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	int count = 0;
	float3 n = make_float3(0,0,0), s = make_float3(0, 0, 0), e = make_float3(0, 0, 0), w = make_float3(0, 0, 0);

	if (!VALID(x, y))
		return;

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
	float3 center = src[IDX(x, y)];

	float3 value = center * count - (n + s + e + w);

	Ap[IDX(x, y)] = value;
}

// out = vector / vector
__global__ void div(const float3* a, const float3* b, float3* out, int size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < size)
		out[idx] = a[idx] / b[idx];
}

// out[n] = a[n] + b[1] * c[n]
__global__ void addmul(float3 *a, const float3* b, const float3 *c, float3* out, int size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= size)
		return;
		
	out[idx] = a[idx] + b[0] * c[idx];
}

// out[n] = a[n] - b[1] * c[n]
__global__ void submul(float3* a, const float3* b, const float3* c, float3* out, int size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= size)
		return;

	out[idx] = a[idx] - b[0] * c[idx];
}

cudaError_t conjugateGradientWithCuda(int numIters, int width, int height, int channels, const float* hostb, float* hostresult)
{
	assert(channels == 3);

	int pixelCount = width * height;
	int pixelDataSize = channels * sizeof(float);
	int dataSize = pixelCount * pixelDataSize;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaSetDevice(0);

	// b x r p,		vector
	// alpha beta	scalar
	// Ap			vector, temp A * p
	// rr			scalar, temp r dot r
	// rrn			scalar, temp rn dot rn
	float3* b = 0;
	float3* x = 0;
	float3* r = 0;
	float3* p = 0;
	float3* alpha = 0;
	float3* beta = 0;
	float3* Ap = 0;
	float3* rr = 0;
	float3* rrn = 0;
	float3* pAp = 0;

	cudaMalloc((void**)&b, dataSize);
	cudaMalloc((void**)&x, dataSize);
	cudaMalloc((void**)&r, dataSize);
	cudaMalloc((void**)&p, dataSize);
	cudaMalloc((void**)&alpha, pixelDataSize);
	cudaMalloc((void**)&beta, pixelDataSize);
	cudaMalloc((void**)&Ap, dataSize);
	cudaMalloc((void**)&rr, pixelDataSize);
	cudaMalloc((void**)&rrn, pixelDataSize);
	cudaMalloc((void**)&pAp, pixelDataSize);

	cudaMemcpy(b, hostb, dataSize, cudaMemcpyHostToDevice);
	// x0 = 0
	cudaMemset(x, 0, dataSize);
	// r0 = b - A * x0 --> r0 = b
	cudaMemcpy(r, b, dataSize, cudaMemcpyDeviceToDevice);
	// p0 = r0
	cudaMemcpy(p, r, dataSize, cudaMemcpyDeviceToDevice);

	CudaCheckError();

	// rr = r ⋅ r
	int blockDim1 = 1024;
	int gridDim1 = (pixelCount + blockDim1 - 1) / blockDim1;
	dot<<<gridDim1, blockDim1, blockDim1*pixelDataSize>>>(r, r, rr, pixelCount); CudaCheckError();


	dim3 blockDim3(32, 32, 1);
	dim3 gridDim3(width, height, 1);
	gridDim3.x = (gridDim3.x + blockDim3.x - 1) / blockDim3.x;
	gridDim3.y = (gridDim3.y + blockDim3.y - 1) / blockDim3.y;

	for (int k = 0; k < numIters; k++) {
		// subscript n mean next = K+1
		// Ap = A * p
		Amulv<<<gridDim3, blockDim3>>>(p, Ap, width, height); CudaCheckError();
		// α = r ⋅ r /(p ⋅ A * p)
		dot<<<gridDim1, blockDim1, blockDim1*pixelDataSize>>>(p, Ap, pAp, pixelCount); CudaCheckError();
		div<<<1,1>>>(rr, pAp, alpha, 1); CudaCheckError();
		// xn = x + α * p
		addmul<<<gridDim1, blockDim1>>>(x, alpha, p, x, pixelCount); CudaCheckError();
		// rn = r - α * A * p
		submul<<<gridDim1, blockDim1>>>(r, alpha, Ap, r, pixelCount); CudaCheckError();
		// β = rn ⋅ rn / r ⋅ r
		dot<<<gridDim1, blockDim1, blockDim1*pixelDataSize>>>(r, r, rrn, pixelCount); CudaCheckError();
		div<<<1,1>>>(rrn, rr, beta, 1); CudaCheckError();
		// pn = rn + β * p
		addmul<<<gridDim1, blockDim1>>>(r, beta, p, p, pixelCount); CudaCheckError();

		std::swap(rr, rrn);
		CudaCheckError();
	}

	cudaDeviceSynchronize();
	cudaMemcpy(hostresult, x, dataSize, cudaMemcpyDeviceToHost);

	CudaCheckError();

	cudaFree(b);
	cudaFree(x);
	cudaFree(r);
	cudaFree(p);
	cudaFree(alpha);
	cudaFree(beta);
	cudaFree(Ap);
	cudaFree(rr);
	cudaFree(rrn);
	cudaFree(pAp);

	return cudaSuccess;

}