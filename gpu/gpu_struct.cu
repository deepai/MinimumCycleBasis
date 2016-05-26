#include "common.cuh"

void gpu_struct::init_memory_setup() {
	CudaError(cudaMalloc(&d_non_tree_edges, to_byte_32bit(num_edges)));
	CudaError(
			cudaMalloc(&d_edge_offsets,
					to_byte_32bit(chunk_size * original_nodes)));
	CudaError(
			cudaMalloc(&d_row_offset,
					to_byte_32bit(chunk_size * (original_nodes + 1))));
	CudaError(
			cudaMalloc(&d_columns, to_byte_32bit(chunk_size * original_nodes)));
	CudaError(
			cudaMalloc(&d_precompute_array,
					to_byte_32bit(chunk_size * original_nodes)));
	CudaError(cudaMalloc(&d_si_vector, to_byte_64bit(size_vector)));
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

void gpu_struct::clear_memory() {
	CudaError(cudaFree(d_non_tree_edges));
	CudaError(cudaFree(d_edge_offsets));
	CudaError(cudaFree(d_row_offset));
	CudaError(cudaFree(d_columns));
	CudaError(cudaFree(d_precompute_array));
	CudaError(cudaFree(d_si_vector));

	destroy_streams();
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

	variable_memory_bytes += to_byte_32bit(chunk_size * original_nodes);
	variable_memory_bytes += to_byte_32bit(chunk_size * (original_nodes + 1));
	variable_memory_bytes += to_byte_32bit(chunk_size * original_nodes);
	variable_memory_bytes += to_byte_32bit(chunk_size * original_nodes);
	variable_memory_bytes += to_byte_32bit(fvs_size);

	total_memory_bytes += static_memory_bytes + variable_memory_bytes;

	printf("Static Memory = %lf mb\n", static_memory_bytes / size_in_mb);
	printf("Variable Memory = %lf mb\n", variable_memory_bytes / size_in_mb);
	printf("total_memory_bytes = %lf mb\n", total_memory_bytes / size_in_mb);

}

void gpu_struct::initialize_memory(gpu_task *host_memory) {
	CudaError(
			cudaMemcpy(d_non_tree_edges, host_memory->non_tree_edges_array,
					to_byte_32bit(num_edges), cudaMemcpyHostToDevice));

	CudaError(
			cudaMemcpy(d_edge_offsets, host_memory->host_tree->edge_offset[0],
					to_byte_32bit(chunk_size * original_nodes),
					cudaMemcpyHostToDevice));

	CudaError(
			cudaMemcpy(d_row_offset, host_memory->host_tree->tree_rows[0],
					to_byte_32bit(chunk_size * (original_nodes + 1)),
					cudaMemcpyHostToDevice));

	CudaError(
			cudaMemcpy(d_columns, host_memory->host_tree->tree_cols[0],
					to_byte_32bit(chunk_size * original_nodes),
					cudaMemcpyHostToDevice));
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

	CudaError(
			cudaMemcpy(host_memory->host_tree->precompute_value[0],
					d_precompute_array,
					to_byte_32bit(chunk_size * original_nodes),
					cudaMemcpyDeviceToHost));

	timer.Stop();

	return timer.Elapsed();
}

void gpu_struct::transfer_from_asynchronous(int start, int end, int fvs_index,
		int stream_index, gpu_task *host_memory) {
	CudaError(
			cudaMemcpyAsync(
					d_edge_offsets + stream_index * chunk_size * original_nodes,
					host_memory->host_tree->edge_offset[fvs_index],
					to_byte_32bit(
							min(end - start, chunk_size) * original_nodes),
					cudaMemcpyHostToDevice, streams[stream_index]));

	CudaError(
			cudaMemcpyAsync(
					d_row_offset
							+ stream_index * chunk_size * (original_nodes + 1),
					host_memory->host_tree->tree_rows[fvs_index],
					to_byte_32bit(
							min(end - start, chunk_size)
									* (original_nodes + 1)),
					cudaMemcpyHostToDevice, streams[stream_index]));

	CudaError(
			cudaMemcpyAsync(
					d_columns + stream_index * chunk_size * original_nodes,
					host_memory->host_tree->tree_cols[fvs_index],
					to_byte_32bit(
							min(end - start, chunk_size) * original_nodes),
					cudaMemcpyHostToDevice, streams[stream_index]));
}

void gpu_struct::transfer_to_asynchronous(int start, int end, int fvs_index,
		int stream_index, gpu_task *host_memory) {
	CudaError(
			cudaMemcpyAsync(host_memory->host_tree->precompute_value[fvs_index],
					d_precompute_array
							+ stream_index * chunk_size * original_nodes,
					to_byte_32bit(
							min(end - start, chunk_size) * original_nodes),
					cudaMemcpyDeviceToHost, streams[stream_index]));
}
