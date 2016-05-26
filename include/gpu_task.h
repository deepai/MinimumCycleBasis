#ifndef _H_GPU_TASK
#define _H_GPU_TASK

#include <vector>

#include "compressed_trees.h"
#include "bit_vector.h"

struct gpu_task {
	int fvs_size;
	int original_nodes;
	int num_non_tree_edges;
	int edge_size;

	int *fvs_array;
	int *non_tree_edges_array;
	compressed_trees *host_tree;
	bit_vector **support_vectors;

	gpu_task(compressed_trees *ht, int *fvs,
			std::vector<int> &non_tree_edges_map, bit_vector **s_vectors,
			int num_non_tree) {
		fvs_array = fvs;
		host_tree = ht;
		fvs_size = ht->fvs_size;
		original_nodes = ht->parent_graph->Nodes;
		num_non_tree_edges = num_non_tree;
		edge_size = non_tree_edges_map.size();
		non_tree_edges_array = non_tree_edges_map.data();
		support_vectors = s_vectors;
	}

	~gpu_task() {

	}
};

#endif
