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


__device__ void multi_search(const int* R,const int* C,const int* fvs_vertices,
							 const int &n,int *d,int *Q, int *Q2,pitch *p,
							 const int &length,const int &stream_index)
{
	int j = threadIdx.x;  //threadId
	int lane_id = getLaneId();     //lane id;
	int warp_id = threadIdx.x/32;

	int src_index;
	int start = stream_index * length;
	int end = start + length;

	__shared__ int *Q_row;
	__shared__ int *Q2_row;

	if(j == 0)
	{
		Q_row = get_row(Q,p->Q_pitch);
		Q2_row = get_row(Q2,p->Q2_pitch);
	}

	__syncthreads();

	for(src_index=blockIdx.x + start; src_index < end; src_index+=gridDim.x)
	{
		int i = __ldg(&fvs_vertices[src_index]);
		int v_discovered = 0;

		int *d_row = get_pointer(d,src_index - start,n,length,stream_index);

		const int* __restrict__ r_row = get_pointer_const(R,src_index - start,n + 1,length,stream_index);
		const int* __restrict__ c_row = get_pointer_const(C,src_index - start,n,length,stream_index);

		__shared__ int Q_len;
		__shared__ int Q2_len;
		__shared__ bool degree_zero_source;

		if(j == 0)
		{
			d_row[i] = 0;

			Q_row[0] = i;
			Q_len = 1;

			Q2_len = 0;

			if(__ldg(&r_row[i+1]) == __ldg(&r_row[i])) //human-readable: if(outdegree == 0)
			{
				degree_zero_source = true;
			}
			else
			{
				degree_zero_source = false;
			}
		}
		__syncthreads();

		//Don't waste time traversing vertices of degree zero
		if(degree_zero_source)
		{
			continue;
		}

		__shared__ int next_queue_element;

		while(1) //While a frontier exists for this source vertex...
		{
			int k = warp_id; //current_queue_element

			//number of warps. Warps [0,w-1] process queue elements [0,w-1] in the current frontier and asynchronously grab elements [w,Q_len).
			if(j == 0)
			{
				next_queue_element = blockDim.x/WARP_SIZE;
			}
			__syncthreads();
			//Let each warp be assigned to an element in the queue, once that element is processed the warp grabs the next queue element, if any. Warps synchronize once all queue elements are handled.

			while(k < Q_len) //Some warps will execute this loop, some won't. When a warp does, all threads in the warp do.
			{
				v_discovered = lane_id;

				int v,r,r_end,d_row_v,old_offset;

				if(lane_id == 0)
				{
					v = Q_row[k];
					r = __ldg(&r_row[v]);
					r_end = __ldg(&r_row[v+1]);
					d_row_v = d_row[v];
					old_offset = atomicAdd(&Q2_len,r_end - r);
				}

				v = __shfl(v,0); //copy from lane 0
				r = __shfl(r,0) + lane_id; //copy from lane 0
				r_end = __shfl(r_end,0); //copy from lane 0
				d_row_v = __shfl(d_row_v,0); //copy from lane 0
				old_offset = __shfl(old_offset,0); //copy from lane 0

				while(r < r_end) //Only some threads in each warp will execute this loop
				{
					int w = __ldg(&c_row[r]);

					//atomics are only needed here when we're computing shortest path calculations
					d_row[w] = d_row_v ^ d_row[w]; //HERE WE UPDATE THE EDGE ENDPOINTS
					r += WARP_SIZE;

					Q2_row[old_offset + v_discovered] = w;

					//increase count by WARP_SIZE
					v_discovered += WARP_SIZE;
				}

				if(lane_id == 0)
				{
					k = atomicAdd(&next_queue_element,1); //Grab the next item off of the queue
				}

				k = __shfl(k,0); //All threads in the warp need the value of k
			}
			__syncthreads();

			//TODO: Combine getMax, insertStack, and updateEndpoints into one functon that resets the queue
			if(Q2_len == 0)
				break;
			else
			{
				if(j == 0)
				{
					int *tmp = Q_row;
					Q_row = Q2_row;
					Q2_row = tmp;

					Q_len = Q2_len;
					Q2_len = 0;
				}

				__syncthreads();
			}
		}

		// __syncthreads(); //debarshi
	}
}
__global__
void __kernel_multi_search_shuffle_based(const int *R,const int *C,const int *fvs_vertices,const int n,int *d,
										 int *Q, int *Q2,pitch *p,
										 const int length,const int stream_index)
{
	//Since users need to handle this, we can provide default policies or clean up the Queueing interfac
	multi_search(R,C,fvs_vertices,n,d,Q,Q2,p,length,stream_index);
}


float gpu_struct::Kernel_multi_search_helper(int start,int end,int stream_index)
{
	assert(end > start);

	int total_length = end - start;

	//debug("multi",start,end);

	timer.Start();

	__kernel_multi_search_shuffle_based<<<min(dimGrid.x,total_length),dimBlock,0,streams[stream_index]>>>(d_row_offset,
																										  d_columns,
																										  d_fvs_vertices,
																										  original_nodes,
																										  d_precompute_array,
																										  Q_d,
																										  Q2_d,
																										  gpu_pitch,
																										  total_length,
																										  stream_index);

	CudaError(cudaStreamSynchronize(streams[stream_index]));

	timer.Stop();

	CudaError(cudaGetLastError());

	float time_elapsed = timer.Elapsed();

	return time_elapsed;
}
