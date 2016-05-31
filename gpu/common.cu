#include "common.cuh"
#include "utils.h"

cudaDeviceProp prop;
int device_id;
dim3 dimGrid;
dim3 dimBlock;

extern "C" void init_cuda() {
	CudaError(cudaGetDevice(&device_id));
	CudaError(cudaGetDeviceProperties(&prop, device_id));
}

extern "C" size_t configure_grid(int start, int end) {
	dimGrid.x = prop.multiProcessorCount;
	dimGrid.y = 1;
	dimGrid.z = 1;

	dimBlock.x = prop.maxThreadsPerBlock;
	dimBlock.y = 1;
	dimBlock.z = 1;

	size_t sources_to_store;
	if ((end - start) < dimGrid.x)
		sources_to_store = end - start;
	else
		sources_to_store = dimGrid.x;

	return sources_to_store;
}

extern "C" unsigned *allocate_pinned_memory(int chunk, int nodes) {
	unsigned *pinned_memory;

	CudaError(
			cudaMallocHost((void **) &pinned_memory,
					sizeof(unsigned) * chunk * nodes));
	//pinned_memory = new unsigned[chunk * nodes];

	return pinned_memory;
}

extern "C" void free_pinned_memory(unsigned *pinned_memory) {
	CudaError(cudaFreeHost(pinned_memory));
	//delete[] pinned_memory;
}

extern "C" int calculate_chunk_size(int num_nodes, int num_edges,
		int size_vector, int nstream) {

}
