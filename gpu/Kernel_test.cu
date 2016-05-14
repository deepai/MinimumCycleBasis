#include "common.cuh"

__global__
void temp_kernel(int a,int b)
{
	int tid = threadIdx.x;
}

void func(int a,int b)
{
	temp_kernel<<<32,1024>>>(a,b);

	cudaDeviceSynchronize();
}