#include "gpu_struct.cuh"
#include "common.cuh"

//Get block of data from pitched pointer and pitch size
template <typename T>
__device__ __forceinline__
T* get_row(T* data, size_t p)
{
	return (T*)((char*)data + blockIdx.x*p);
}

template <typename T>
__device__ __forceinline__
T* get_pointer(T* data,int node_index,int num_nodes,int chunk_size,int stream_id)
{
	return (data + (stream_id * chunk_size * num_nodes) + (node_index * num_nodes));
}

template <typename T>
__device__ __forceinline__
const T* get_pointer_const(const T* data,int node_index,int num_nodes,int chunk_size,int stream_id)
{
	return (data + (stream_id * chunk_size * num_nodes) + (node_index * num_nodes));
}

__device__ __forceinline__
unsigned getBit(unsigned long long val, int pos)
{
	unsigned long long ret;
	asm("bfe.u64 %0, %1, %2, 1;" : "=l"(ret) : "l"(val), "r"(pos));
	return	(unsigned)ret;
}

//return vertex outdegree
__device__ __forceinline__
int outdegree(int v, const int *R)
{
	return __ldg(&R[v+1])-__ldg(&R[v]);
}

__device__ __forceinline__
int getWarpId()
{
	return (threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x)/WARP_SIZE;
}

//Returns the current thread's lane ID
__device__ __forceinline__
int getLaneId()
{
	int lane_id;
	asm("mov.s32 %0, %laneid;" : "=r"(lane_id));
	return lane_id;
}

// Insert a single bit into 'val' at position 'pos'
__device__ __forceinline__
unsigned setBit(unsigned val, unsigned toInsert, int pos)
{
	unsigned ret;
	asm("bfi.b32 %0, %1, %2, %3, 1;" : "=r"(ret) : "r"(toInsert), "r"(val), "r"(pos));
	return ret;
}


#define Q_THRESHOLD 0


__device__ void multi_search(const int* R,const int* C,const int &n,int *d,
			     const int &length,const int &stream_index)
{
	int j = threadIdx.x;  //threadId
	int lane_id = getLaneId();     //lane id;
	int warp_id = threadIdx.x/32;

	int src_index;

	int start = stream_index * length;
	int end = start + length;

	for(src_index=blockIdx.x*WARP_SIZE + warp_id + start; src_index < end; src_index += (gridDim.x)*WARP_SIZE)
	{
		int *d_row = get_pointer(d,src_index - start,n,length,stream_index);

		const int* __restrict__ r_row = get_pointer_const(R,src_index - start,n + 1,length,stream_index);
		const int* __restrict__ c_row = get_pointer_const(C,src_index - start,n,length,stream_index);

		int k = 0; //current level

		int r,r_end,r_prev;

		while(k < n) //All threads in a warp simultaneously executes nodes in a level.
		{
			if(lane_id == 0)
			{
				r_prev = __ldg(&r_row[k]);
				r_end   = __ldg(&r_row[k+1]);
			}

			r_prev = __shfl(r_prev,0);
			r_end = __shfl(r_end,0);
			r = r_prev + lane_id;

			while(r < r_end)
			{
				int c = __ldg(&c_row[r]); //c is the index of the parent of the current edge. if c == -1, its the root node

				if(c != -1)
					d_row[r] = d_row[r] ^ d_row[c];

				r += WARP_SIZE;
			}

			if(r_prev == r_end)
				break;

			k++;
		}
	}
}
__global__
void __kernel_multi_search_shuffle_based(const int *R,const int *C,const int n,int *d,const int length,
					 const int stream_index)
{
	//Since users need to handle this, we can provide default policies or clean up the Queueing interfac
	multi_search(R,C,n,d,length,stream_index);
}


float gpu_struct::Kernel_multi_search_helper(int start,int end,int stream_index)
{
	assert(end > start);

	int total_length = end - start;

	//debug("multi",start,end);

	timer.Start();

	__kernel_multi_search_shuffle_based<<<min(dimGrid.x,total_length),dimBlock,0,streams[stream_index]>>>(d_row_offset,
														d_columns,
														original_nodes,
														d_precompute_array,
														total_length,
														stream_index);

	CudaError(cudaStreamSynchronize(streams[stream_index]));

	timer.Stop();

	CudaError(cudaGetLastError());

	float time_elapsed = timer.Elapsed();

	return time_elapsed;
}
