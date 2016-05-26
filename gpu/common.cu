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

extern "C" size_t calculate_chunk_size(size_t num_nodes, size_t num_edges,
		size_t size_vector, size_t nstream) {
	size_t global_storage_bytes = prop.totalGlobalMem;
	size_t static_storage_bytes = calculate_32bit(
			num_edges) + calculate_64bit(size_vector);

	size_t remaining_storage_bytes = global_storage_bytes
			- static_storage_bytes;
	size_t total_elem_avl = remaining_storage_bytes / 4;

	size_t max_chunk_size = total_elem_avl / (nstream * (num_nodes * 4 + 1));

	debug("global_storage_bytes", global_storage_bytes);
	debug("static_storage_bytes", static_storage_bytes);
	debug("remaining_storage_bytes", remaining_storage_bytes);
	debug("total_elem_avl", total_elem_avl);
	debug("max_chunk_size", max_chunk_size);

	return max_chunk_size;
}
