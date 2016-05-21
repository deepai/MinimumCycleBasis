#include "gpu_struct.cuh"
#include "common.cuh"
#include "pitch.h"


template <typename T>
__device__ __forceinline__
const T* get_pointer_const(const T* data,int node_index,int num_nodes,int chunk_size,int stream_id)
{
	return (data + (stream_id * chunk_size * num_nodes) + (node_index * num_nodes));
}


template <typename T>
__device__ __forceinline__
T* get_pointer(T* data,int node_index,int num_nodes,int chunk_size,int stream_id)
{
	return (data + (stream_id * chunk_size * num_nodes) + (node_index * num_nodes));
}

__device__ __forceinline__
int get_ceil(float dividend,int divisor)
{
	return ((int)(ceilf(dividend/divisor)));
}

__device__ __forceinline__
unsigned getBit(unsigned long long val, int pos)
{
	unsigned long long ret;
	asm("bfe.u64 %0, %1, %2, 1;" : "=l"(ret) : "l"(val), "r"(pos));
	return	(unsigned)ret;
}

__global__
void __kernel_init_edge(const int* __restrict__ d_non_tree_edges,const int* d_edge_offsets,
						int *d_precompute_array,const int* __restrict__ d_fvs_vertices,
						const unsigned long long *d_si_vector,int start,int end,
						int stream_index,int chunk_size,int original_nodes,int size_vector,
						int fvs_size,int num_non_tree_edges,int num_edges)
{
	int tid = threadIdx.x;
	int gid = threadIdx.x + blockDim.x*blockIdx.x;

	int si_index = -1;

	unsigned long long si_value;

	//int per_sm_vtx_count = get_ceil(end - start,gridDim.x);

	for(int src_index=blockIdx.x + start; src_index < end; src_index+=gridDim.x)
	{
		assert(src_index < fvs_size && src_index < end);

		int src = __ldg(&d_fvs_vertices[src_index]);

		int *d_row = get_pointer(d_precompute_array,src_index,original_nodes,chunk_size,stream_index);
		const int* __restrict__ d_edge = get_pointer_const(d_edge_offsets,src_index,original_nodes,chunk_size,stream_index);

		for(int edge_index = tid; edge_index < original_nodes ; edge_index += blockDim.x)
		{
			int edge_offset = __ldg(&d_edge[edge_index]);
			assert(edge_offset < num_edges);
			//tree edges
			if(edge_offset >= 0)
			{
				int non_tree_edge_loc = __ldg(&d_non_tree_edges[edge_offset]);
				assert(non_tree_edge_loc < num_non_tree_edges);

				//non_tree_edge
				if(non_tree_edge_loc >= 0)
				{
					int p_idx = non_tree_edge_loc/64;
					if(si_index < 0)
					{
						si_index = p_idx;
						si_value = __ldg(&d_si_vector[si_index]);
					}
					else if(si_index != p_idx)
					{
						si_index = p_idx;
						si_value = __ldg(&d_si_vector[si_index]);
					}

					d_row[edge_index] = getBit(si_value,non_tree_edge_loc%64);
				}
				else //tree edge
					d_row[edge_index] = 0;
			}
			else
			{
				assert(edge_offset == -1);
				d_row[edge_index] = 0;
			}
		}

		__syncthreads();
	}

}

/**
 * @brief This method is used to invoke a kernel whose function is defined in the details section.
 * @details This method invokes a Kernel. The Kernel's task is to parallely do the following things in the order.
 * a)For each source vertex between start and end (15 at a time(grid dimension)). We fill the precompute_array edges
 * b)The precompute array is filled in the following way.
 *   If the edge is a tree edge in the original spanning tree. then its value is 0.
 *   else if Si contains 1 in the corresponding non-tree edge position then 1 else 0.
 *
 * @param start index of vertex from 0 - fvs_size - 2
 * @param end index of vertex from 1 to fvs_size - 1
 * @param stream_index 0 or 1
 */
float gpu_struct::Kernel_init_edges_helper(int start,int end,int stream_index)
{
	assert(end > start);

	int total_length = end - start;

	timer.Start();

	__kernel_init_edge<<<min(dimGrid.x,total_length),dimBlock,0,streams[stream_index]>>>(d_non_tree_edges,
																					 d_edge_offsets,
																					 d_precompute_array,
																					 d_fvs_vertices,
																					 d_si_vector,
																					 start,
																					 end,
																					 stream_index,
																					 chunk_size,
																					 original_nodes,
																					 size_vector,
																					 fvs_size,
																					 num_non_tree_edges,
																					 num_edges);

	CudaError(cudaStreamSynchronize(streams[stream_index]));

	timer.Stop();

	CudaError(cudaGetLastError());

	float time_elapsed = timer.Elapsed();

	return time_elapsed;
}
