#ifndef __H_COMMON_CUDA
#define __H_COMMON_CUDA

#include "gpu_struct.cuh"

#define calculate_8bit(X) (X)
#define calculate_16bit(X) (2*X)
#define calculate_32bit(X) (4*X)
#define calculate_64bit(X) (8*X)

#define WARP_SIZE 32

extern cudaDeviceProp prop;
extern int device_id;
extern dim3 dimGrid;
extern dim3 dimBlock;

extern "C" void init_cuda();

extern "C" size_t configure_grid(int start, int end);

extern "C" unsigned* allocate_pinned_memory(int chunk, int nodes);

extern "C" void free_pinned_memory(unsigned *);

extern "C" int calculate_chunk_size(int num_nodes, int num_edges,
		int size_vector, int nstream);

#endif
