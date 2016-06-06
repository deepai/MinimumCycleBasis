#include "common.cuh"

void gpu_struct::init_memory_setup() {
	CudaError(cudaMalloc(&d_non_tree_edges, to_byte_32bit(num_edges)));
	CudaError(
			cudaMalloc(&d_edge_offsets,
					to_byte_32bit(chunk_size * original_nodes * nstreams)));
	CudaError(
			cudaMalloc(&d_row_offset,
					to_byte_32bit(
							chunk_size * (original_nodes + 1) * nstreams)));
	CudaError(cudaMalloc(&d_columns,
	to_byte_32bit(chunk_size * original_nodes) * nstreams));
	CudaError(cudaMalloc(&d_precompute_array,
	to_byte_32bit(chunk_size * original_nodes) * nstreams));
	CudaError(cudaMalloc(&d_si_vector, to_byte_64bit(size_vector)));
}

void gpu_struct::clear_memory() {
	CudaError(cudaFree(d_non_tree_edges));
	CudaError(cudaFree(d_edge_offsets));
	CudaError(cudaFree(d_row_offset));
	CudaError(cudaFree(d_columns));
	CudaError(cudaFree(d_precompute_array));
	CudaError(cudaFree(d_si_vector));

	destroy_streams();
}

void gpu_struct::init_streams() {
	streams = (cudaStream_t*) malloc(nstreams * sizeof(cudaStream_t));
	for (int i = 0; i < nstreams; i++)
		CudaError(cudaStreamCreate(&(streams[i])));
}

void gpu_struct::destroy_streams() {
	for (int i = 0; i < nstreams; i++)
		CudaError(cudaStreamDestroy(streams[i]));

	free(streams);
}

void gpu_struct::init_pitch() {
}

void gpu_struct::calculate_memory() {
	int total_memory_bytes = 0;
	int static_memory_bytes = 0;
	int variable_memory_bytes = 0;

	float size_in_mb = 1024 * 1024;

	static_memory_bytes += to_byte_32bit(num_edges);
	static_memory_bytes += to_byte_64bit(size_vector);

	variable_memory_bytes += to_byte_32bit(
			chunk_size * original_nodes * nstreams);
	variable_memory_bytes += to_byte_32bit(
			chunk_size * (original_nodes + 1) * nstreams);
	variable_memory_bytes += to_byte_32bit(
			chunk_size * original_nodes * nstreams);
	variable_memory_bytes += to_byte_32bit(
			chunk_size * original_nodes * nstreams);

	total_memory_bytes += static_memory_bytes + variable_memory_bytes;

	printf("Static Memory = %lf mb\n", static_memory_bytes / size_in_mb);
	printf("Variable Memory = %lf mb\n", variable_memory_bytes / size_in_mb);
	printf("total_memory_bytes = %lf mb\n", total_memory_bytes / size_in_mb);

}

void gpu_struct::initialize_memory(gpu_task *host_memory) {

	CudaError(
			cudaMemcpy(d_non_tree_edges, host_memory->non_tree_edges_array,
					to_byte_32bit(num_edges), cudaMemcpyHostToDevice));

	for (int i = 0; i < nstreams; i++) {
		CudaError(
				cudaMemcpy(d_edge_offsets + chunk_size * original_nodes * i,
						host_memory->host_tree->edge_offset[i],
						to_byte_32bit(chunk_size * original_nodes),
						cudaMemcpyHostToDevice));

		CudaError(
				cudaMemcpy(d_row_offset + chunk_size * (original_nodes + 1) * i,
						host_memory->host_tree->tree_rows[i],
						to_byte_32bit(chunk_size * (original_nodes + 1)),
						cudaMemcpyHostToDevice));

		CudaError(
				cudaMemcpy(d_columns + chunk_size * original_nodes * i,
						host_memory->host_tree->tree_cols[i],
						to_byte_32bit(chunk_size * original_nodes),
						cudaMemcpyHostToDevice));
	}
}

float gpu_struct::copy_support_vector(bit_vector *vector) {
	timer.Start();

	CudaError(
			cudaMemcpy(d_si_vector, vector->elements,
					to_byte_64bit(size_vector), cudaMemcpyHostToDevice));

	timer.Stop();

	return timer.Elapsed();
}

float gpu_struct::fetch(gpu_task *host_memory) {
	timer.Start();

	for (int i = 0; i < nstreams; i++) {
		CudaError(
				cudaMemcpy(host_memory->host_tree->precompute_value[i],
						d_precompute_array + chunk_size * original_nodes * i,
						to_byte_32bit(chunk_size * original_nodes),
						cudaMemcpyDeviceToHost));
	}

	timer.Stop();

	return timer.Elapsed();
}

void gpu_struct::transfer_from_asynchronous(int stream_index,
		gpu_task *host_memory,int num_chunk) {

	CudaError(
			cudaMemcpyAsync(
					d_edge_offsets + stream_index * chunk_size * original_nodes,
					host_memory->host_tree->edge_offset[num_chunk],
					to_byte_32bit(chunk_size * original_nodes),
					cudaMemcpyHostToDevice, streams[stream_index]));

	CudaError(
			cudaMemcpyAsync(
					d_row_offset
							+ stream_index * chunk_size * (original_nodes + 1),
					host_memory->host_tree->tree_rows[num_chunk],
					to_byte_32bit(chunk_size * (original_nodes + 1)),
					cudaMemcpyHostToDevice, streams[stream_index]));

	CudaError(
			cudaMemcpyAsync(
					d_columns + stream_index * chunk_size * original_nodes,
					host_memory->host_tree->tree_cols[num_chunk],
					to_byte_32bit(chunk_size * original_nodes),
					cudaMemcpyHostToDevice, streams[stream_index]));
}

void gpu_struct::transfer_to_asynchronous(int stream_index,
		gpu_task *host_memory,int num_chunk) {
	CudaError(
			cudaMemcpyAsync(
					host_memory->host_tree->precompute_value[num_chunk],
					d_precompute_array
							+ stream_index * chunk_size * original_nodes,
					to_byte_32bit(chunk_size * original_nodes),
					cudaMemcpyDeviceToHost, streams[stream_index]));
}

float gpu_struct::process_shortest_path(gpu_task *host_memory,
		bool multiple_transfer) {
	timer.Start();

	for (int i = 0; i < num_chunks; i++) {

		int start = (i%nstreams) * chunk_size;
		int end = (i%nstreams + 1) * chunk_size;

		if (multiple_transfer)
			transfer_from_asynchronous(i%nstreams, host_memory, i);

		Kernel_init_edges_helper(start, end, i%nstreams);

		Kernel_multi_search_helper(start, end, i%nstreams);

		transfer_to_asynchronous(i%nstreams, host_memory, i);
	}

	CudaError(cudaDeviceSynchronize());

	timer.Stop();

	return timer.Elapsed();
}
