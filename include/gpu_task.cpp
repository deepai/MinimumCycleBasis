#include "gpu_task.h"

int gpu_task::calculate_fixed_storage()
{
	int total_byes = 0;
	total_byes += calculate_32bit(num_non_tree_edges);
	total_byes += calculate_64bit(support_vectors[0]->size);

	return total_byes;
}

int gpu_task::calculate_variable_storage()
{
	int total_bytes = 0;
	total_bytes += (7 * calculate_32bit(original_nodes));
	total_bytes += (calculate_32bit(original_nodes + 1));

	return total_bytes;
}

void gpu_task::initialize()
{
	int fixed_memory_in_bytes = calculate_fixed_storage();
	int variable_memory_in_bytes = calculate_variable_storage();

	float memory_in_mb = 1024*1024;

	#ifdef PRINT
		printf("Memory Requirements =======>>> \nFixed Storage = %f mb and Variable Storage = %f mb \n", fixed_memory_in_bytes/memory_in_mb,
																	variable_memory_in_bytes/memory_in_mb );
	#endif
}
