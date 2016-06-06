#ifndef __H_GPU_STRUCT
#define __H_GPU_STRUCT

#include "utils.h"
#include "gpu_task.h"
#include "bit_vector.h"
#include "gputimer.h"
#include "stats.h"

#include <stdio.h>

#define to_byte_32bit(X) (X * sizeof(int))
#define to_byte_64bit(X) (X * sizeof(long long))

struct gpu_struct {
	int num_edges;
	int size_vector;
	int original_nodes;
	int fvs_size;
	int chunk_size;
	int num_chunks;
	int num_non_tree_edges;

	int *d_non_tree_edges;
	int *d_edge_offsets;
	int *d_row_offset;
	int *d_columns;
	int *d_precompute_array;
	int *d_fvs_vertices;

	int *d_counter;

	GpuTimer timer;

	//Device pointers for queues
	int nstreams;

	cudaStream_t* streams;

	stats *info;

	unsigned long long *d_si_vector;

	gpu_struct(int num_edges, int num_non_tree_edges, int size_vector,
			int original_nodes, int fvs_size, int chunk_size, int nstreams, stats *info) {
		this->num_non_tree_edges = num_non_tree_edges;
		this->num_edges = num_edges;
		this->size_vector = size_vector;
		this->original_nodes = original_nodes;
		this->fvs_size = fvs_size;
		this->chunk_size = chunk_size;
		this->nstreams = std::min(32, nstreams);
		this->num_chunks = nstreams;
		this->info = info;

		init_memory_setup();
		init_pitch();
		init_streams();
	}

	void init_memory_setup();
	void init_pitch();

	void init_streams();
	void destroy_streams();

	void calculate_memory();

	void initialize_memory(gpu_task *host_memory);
	float copy_support_vector(bit_vector *vector);

	void transfer_from_asynchronous(int stream_index, gpu_task *host_memory,int num_chunk);

	float fetch(gpu_task *host_memory);

	void transfer_to_asynchronous(int stream_index, gpu_task *host_memory,int num_chunk);

	void Kernel_init_edges_helper(int start, int end, int stream_index);
	void Kernel_multi_search_helper(int start, int end, int stream_index);

	void clear_memory();

	float process_shortest_path(gpu_task *host_memory,bool multiple_transfer);
};

#endif
