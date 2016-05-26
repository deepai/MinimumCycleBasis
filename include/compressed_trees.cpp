#include "compressed_trees.h"

int compressed_trees::get_node_arrays(unsigned **csr_rows, unsigned **csr_cols,
		int **csr_edge_offset, int **csr_parent, int **csr_distance,
		int node_index) {
	int row_number = node_index / chunk_size;
	int col_number = node_index % chunk_size;

	*csr_rows = tree_rows[row_number] + (col_number * (original_nodes + 1));
	*csr_cols = tree_cols[row_number] + (col_number * original_nodes);

	*csr_edge_offset = edge_offset[row_number] + (col_number * original_nodes);
	*csr_parent = parent[row_number] + (col_number * original_nodes);

	*csr_distance = distance[row_number] + (col_number * original_nodes);
}

int compressed_trees::get_node_arrays_warp(unsigned **csr_rows,
		unsigned **csr_cols, int **csr_edge_offset, int **csr_parent,
		int **csr_distance, unsigned **csr_nodes_index, int node_index) {
	int row_number = node_index / chunk_size;
	int col_number = node_index % chunk_size;

	*csr_rows = tree_rows[row_number] + (col_number * (original_nodes + 1));
	*csr_cols = tree_cols[row_number] + (col_number * original_nodes);

	*csr_edge_offset = edge_offset[row_number] + (col_number * original_nodes);
	*csr_parent = parent[row_number] + (col_number * original_nodes);

	*csr_distance = distance[row_number] + (col_number * original_nodes);
	*csr_nodes_index = nodes_index[row_number] + (col_number * original_nodes);
}

int compressed_trees::get_precompute_array(unsigned **precompute_tree,
		int node_index) {
	int row_number = node_index / chunk_size;
	int col_number = node_index % chunk_size;

	*precompute_tree = precompute_value[row_number]
			+ (col_number * original_nodes);
}

int compressed_trees::get_index(int original_node) {
	return vertices_map[original_node];
}

void compressed_trees::copy(int index, std::vector<unsigned> *tree_edges,
		std::vector<int> *parent_edges, std::vector<int> *distances) {
	assert(index < fvs_size);

	int row_number = index / chunk_size;
	int col_number = index % chunk_size;

	unsigned src = final_vertices[index];

	unsigned row, col, offset;

	parent[row_number][col_number * original_nodes + src] = -1;
	distance[row_number][col_number * original_nodes + src] = 0;

	for (int i = 0; i < parent_edges->size(); i++) {
		offset = parent_edges->at(i);

		if (offset == -1) {
			assert(i == src);
			continue;
		}

		col = parent_graph->columns->at(offset);

		assert(col == i);

		parent[row_number][col_number * original_nodes + i] = offset;
		distance[row_number][col_number * original_nodes + i] = distances->at(
				i);

	}

	unsigned *node_rowoffsets, *node_columns;
	int *node_edgeoffsets, *node_parents, *node_distance;

	get_node_arrays(&node_rowoffsets, &node_columns, &node_edgeoffsets,
			&node_parents, &node_distance, index);

	parent_graph->fill_tree_edges(node_rowoffsets, node_columns,
			node_edgeoffsets, tree_edges, src);

}

void compressed_trees::print_tree() {
	for (int i = 0; i < chunk_size; i++) {
		printf("src = %d\n", final_vertices[i] + 1);
		for (int j = 0; j < original_nodes; j++) {
			int edge_offset = parent[0][i * original_nodes + j];
			if (edge_offset != -1) {
				printf("%d - %d, %d\n", parent_graph->rows->at(edge_offset) + 1,
						parent_graph->columns->at(edge_offset) + 1,
						precompute_value[0][i * original_nodes + j]);
			}
		}
		printf(
				"===============================================================================\n");
	}
}
