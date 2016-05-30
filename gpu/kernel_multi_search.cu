#include "gpu_struct.cuh"
#include "common.cuh"

//Get block of data from pitched pointer and pitch size
template<typename T>
__device__  __forceinline__ T* get_row(T* data, size_t p) {
	return (T*) ((char*) data + blockIdx.x * p);
}

template<typename T>
__device__  __forceinline__ T* get_pointer(T* data, int node_index,
		int num_nodes, int chunk_size, int stream_id) {
	return (data + (stream_id * chunk_size * num_nodes)
			+ (node_index * num_nodes));
}

template<typename T>
__device__  __forceinline__
 const T* get_pointer_const(const T* data,
		int node_index, int num_nodes, int chunk_size, int stream_id) {
	return (data + (stream_id * chunk_size * num_nodes)
			+ (node_index * num_nodes));
}

__device__ __forceinline__
unsigned getBit(unsigned long long val, int pos) {
	unsigned long long ret;
	asm("bfe.u64 %0, %1, %2, 1;" : "=l"(ret) : "l"(val), "r"(pos));
	return (unsigned) ret;
}

//return vertex outdegree
__device__ __forceinline__
int outdegree(int v, const int *R) {
	return __ldg(&R[v + 1]) - __ldg(&R[v]);
}

__device__ __forceinline__
int getWarpId() {
	return (threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x
			+ threadIdx.x) / WARP_SIZE;
}

//Returns the current thread's lane ID
__device__ __forceinline__
int getLaneId() {
	int lane_id;
	asm("mov.s32 %0, %laneid;" : "=r"(lane_id));
	return lane_id;
}

// Insert a single bit into 'val' at position 'pos'
__device__ __forceinline__
unsigned setBit(unsigned val, unsigned toInsert, int pos) {
	unsigned ret;
	asm("bfi.b32 %0, %1, %2, %3, 1;" : "=r"(ret) : "r"(toInsert), "r"(val), "r"(pos));
	return ret;
}

#define Q_THRESHOLD 0

__global__
void __kernel_multi_search_shuffle_based(const int *R, const int *C,
		const int n, int *d, const int start, const int end,
		const int chunk_size, const int stream_index) {
	int lane_id = getLaneId();     //lane id;

	int src_index = start + blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;

	if (src_index >= end)
		return;

	int *d_row = get_pointer(d, src_index - start, n, chunk_size, stream_index);

	const int* __restrict__ r_row = get_pointer_const(R, src_index - start,
			n + 1, chunk_size, stream_index);
	const int* __restrict__ c_row = get_pointer_const(C, src_index - start, n,
			chunk_size, stream_index);

	int k = 1; //current level

	int r, r_end, r_prev;

	while (k < n) //All threads in a warp simultaneously executes nodes in a level.
	{
		if (lane_id == 0) {
			r_prev = __ldg(&r_row[k]);
			r_end = __ldg(&r_row[k + 1]);
		}

		r_prev = __shfl(r_prev, 0);
		r_end = __shfl(r_end, 0);
		r = r_prev + lane_id;

		while (r < r_end) {
			int c = __ldg(&c_row[r]); //c is the index of the parent of the current edge. if c == -1, its the root node

			d_row[r] = d_row[r] ^ d_row[c];

			r += WARP_SIZE;
		}

		if (r_prev == r_end)
			break;

		k++;
	}
}

void gpu_struct::Kernel_multi_search_helper(int start, int end,
		int stream_index) {

	int total_length = end - start;

	__kernel_multi_search_shuffle_based<<<
			(int) ceil((double) total_length / 16), 512, 0,
			CU_STREAM_PER_THREAD>>>(d_row_offset, d_columns, original_nodes,
			d_precompute_array, start, end, chunk_size, stream_index);

	CudaError(cudaStreamSynchronize(CU_STREAM_PER_THREAD));

	CudaError(cudaGetLastError());
}
