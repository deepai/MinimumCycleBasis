#ifndef _H_WORK_PER_THREAD
#define _H_WORK_PER_THREAD

#include <vector>
#include "CsrTree.h"
#include "cycles.h"
#include "Dijkstra.h"
#include "bit_vector.h"
#include "cycle_searcher.h"
#include "compressed_trees.h"
#include <queue>

#include <utility>
#include <unordered_map>
#include <assert.h>

struct worker_thread {
	dijkstra *helper;

	cycle_storage *storage;

	int *fvs_array;

	compressed_trees *trees;

	std::vector<unsigned> shortest_path_trees;

	worker_thread(csr_multi_graph *graph, cycle_storage *s, int *fvs_array,
			compressed_trees *tr) {
		helper = new dijkstra(graph->Nodes, graph, fvs_array);
		storage = s;
		this->fvs_array = fvs_array;
		trees = tr;
	}

	~worker_thread() {
		delete helper;
		shortest_path_trees.clear();
	}

	int produce_sp_tree_and_cycles(int src_index, csr_multi_graph *graph) {
		assert(src_index >= 0 && src_index < trees->fvs_size);

		int src = trees->final_vertices[src_index];

		assert(src >= 0 && src < graph->Nodes);

		helper->reset();

		csr_tree *sp_tree = new csr_tree(graph);

		//compute shortest path spanning tree and also non-tree edges
		sp_tree->obtain_shortest_path_tree(*helper, true, src);

		//compute the cycles;
		std::vector<unsigned> *non_tree_edges = sp_tree->non_tree_edges;

		int total_weight, temp_weight;
		bool is_edge_cycle, temp_check;

		int count_cycle = 0;

		for (int i = 0; i < non_tree_edges->size(); i++) {
			total_weight = 0;
			is_edge_cycle = helper->is_edge_cycle(non_tree_edges->at(i),
					total_weight, src);

			if (is_edge_cycle) {
				cycle *cle = new cycle(trees, sp_tree->root,
						non_tree_edges->at(i));

				cle->total_length = total_weight;

				storage->add_cycle(src,
						helper->graph->rows->at(non_tree_edges->at(i)),
						helper->graph->columns->at(non_tree_edges->at(i)), cle);

				count_cycle++;
			}
		}

		shortest_path_trees.push_back(src);

		trees->copy(src_index, sp_tree->tree_edges, sp_tree->parent_edges,
				sp_tree->distance);

		delete sp_tree;

		return count_cycle;
	}

	int produce_sp_tree_and_cycles_warp(int src_index, csr_multi_graph *graph) {
		assert(src_index >= 0 && src_index < trees->fvs_size);

		int src = trees->final_vertices[src_index];

		assert(src >= 0 && src < graph->Nodes);

		helper->reset();

		csr_tree *sp_tree = new csr_tree(graph);

		//compute shortest path spanning tree and also non-tree edges
		sp_tree->obtain_shortest_path_tree(*helper, true, src);

		//compute the cycles;
		std::vector<unsigned> *non_tree_edges = sp_tree->non_tree_edges;

		int total_weight, temp_weight;
		bool is_edge_cycle, temp_check;

		int count_cycle = 0;

		for (int i = 0; i < non_tree_edges->size(); i++) {
			total_weight = 0;
			is_edge_cycle = helper->is_edge_cycle(non_tree_edges->at(i),
					total_weight, src);

			if (is_edge_cycle) {
				cycle *cle = new cycle(trees, sp_tree->root,
						non_tree_edges->at(i));

				cle->total_length = total_weight;

				storage->add_cycle(src,
						helper->graph->rows->at(non_tree_edges->at(i)),
						helper->graph->columns->at(non_tree_edges->at(i)), cle);

				count_cycle++;
			}
		}

		shortest_path_trees.push_back(src);

		unsigned *csr_rows, *csr_cols, *csr_nodes_index;
		int *csr_edge_offset, *csr_parent, *csr_distance;

		trees->get_node_arrays_warp(&csr_rows, &csr_cols, &csr_edge_offset,
				&csr_parent, &csr_distance, &csr_nodes_index, src_index);

		helper->fill_tree_edges(csr_rows, csr_cols, csr_nodes_index,
				csr_edge_offset, csr_parent, csr_distance, src);

		delete sp_tree;

		return count_cycle;
	}

	// void empty_cycles()
	// {
	// 	storage->clear_cycles();
	// }

	void precompute_supportVec(std::vector<int> &non_tree_edges,
			bit_vector &vector) {
		//assert(non_tree_edge_map.size() == vector.get_num_elements());
		//assert(vector.get_size() == (int)(ceil((double)non_tree_edge_map.size()/64)));

		for (int i = 0; i < shortest_path_trees.size(); i++) {
			unsigned src = shortest_path_trees[i];
			int src_index = trees->get_index(src);
			unsigned *node_rowoffsets, *node_columns, *precompute_nodes;
			int *node_edgeoffsets, *node_parents, *node_distance;

			trees->get_node_arrays(&node_rowoffsets, &node_columns,
					&node_edgeoffsets, &node_parents, &node_distance,
					src_index);
			trees->get_precompute_array(&precompute_nodes, src_index);

			csr_multi_graph *graph = trees->parent_graph;

			precompute_nodes[src] = 0;

			unsigned edge_offset, reverse_edge, row, column, position, bit;

			std::queue<unsigned> q;

			q.push(src);
			while (!q.empty()) {
				unsigned top_node = q.front();
				q.pop();

				assert(
						precompute_nodes[top_node] == 0
								|| precompute_nodes[top_node] == 1);

				for (int j = node_rowoffsets[top_node];
						j < node_rowoffsets[top_node + 1]; j++) {
					column = node_columns[j];
					q.push(column);

					edge_offset = node_edgeoffsets[j];
					assert(edge_offset < non_tree_edges.size());

					if (non_tree_edges[edge_offset] >= 0) {
						assert(
								non_tree_edges[edge_offset]
										< vector.num_elements);
						bit = vector.get_bit(non_tree_edges[edge_offset]);
						precompute_nodes[column] = (precompute_nodes[top_node]
								+ bit) % 2;
					} else {
						precompute_nodes[column] = precompute_nodes[top_node];
					}

				}
			}
			precompute_nodes[src] = 0;
		}
	}
};

#endif
