#include "CsrGraphMulti.h"

/**
 * @brief This method is used to obtain the spanning tree of a graph. The spanning tree contains edge offsets from the csr_graph
 * @details This method may also return the list of non_tree_edges and ear decomposition corresponding to every non-tree 
 * 	    edges.
 *  
 * @param  address of an vector for storing non_tree_edges,ear decomposition vector;
 * @return vector of edge_offsets in bfs ordering.
 */
std::vector<unsigned> *csr_multi_graph::get_spanning_tree(
		std::vector<unsigned> **non_tree_edges, int src) {
	struct DFS_HELPER {
		int Nodes;

		std::vector<unsigned> *spanning_tree;
		std::vector<bool> *visited;

		std::vector<unsigned char> *is_tree_edge;

		std::vector<unsigned> *rows_internal;
		std::vector<unsigned> *columns_internal;
		std::vector<unsigned> *rowOffsets_internal;

		std::vector<unsigned> **non_tree_edges_internal;

		std::vector<unsigned> *reverse_edge_internal;

		std::vector<int> *parent;

		DFS_HELPER(std::vector<unsigned> **non_tree_edges,
				std::vector<unsigned> *rows, std::vector<unsigned> *columns,
				std::vector<unsigned> *rowOffsets,
				std::vector<unsigned> *reverse_edge, int _nodes) {
			spanning_tree = new std::vector<unsigned>();
			visited = new std::vector<bool>();
			parent = new std::vector<int>();
			is_tree_edge = new std::vector<unsigned char>();

			Nodes = _nodes;
			for (int i = 0; i < Nodes; i++) {
				parent->push_back(-1);
				visited->push_back(false);
			}

			for (int i = 0; i < rows->size(); i++)
				is_tree_edge->push_back(0);

			non_tree_edges_internal = non_tree_edges;

			rows_internal = rows;
			columns_internal = columns;
			rowOffsets_internal = rowOffsets;
			reverse_edge_internal = reverse_edge;
		}

		void dfs(unsigned row) {
			visited->at(row) = true;

			for (unsigned offset = rowOffsets_internal->at(row);
					offset < rowOffsets_internal->at(row + 1); offset++) {
				unsigned column = columns_internal->at(offset);
				if (!visited->at(column)) {
					visited->at(column) = true;
					spanning_tree->push_back(offset);
					parent->at(column) = row;

					is_tree_edge->at(offset) = 1;

					dfs(column);
				} else {
					if (column == parent->at(row)) {
						int reverse_index = reverse_edge_internal->at(offset);
						if (is_tree_edge->at(reverse_index) == 1) {
							is_tree_edge->at(offset) = 1;
							continue;
						} else if (is_tree_edge->at(reverse_index) == 2) {
							is_tree_edge->at(offset) = 2;
							continue;
						}

						else {
							is_tree_edge->at(offset) = 2;
							is_tree_edge->at(reverse_index) = 2;

							if (row < column)
								(*non_tree_edges_internal)->push_back(offset);
							else
								(*non_tree_edges_internal)->push_back(
										reverse_index);

							continue;
						}
					} else if (is_tree_edge->at(offset) == 0) {
						int reverse_index = reverse_edge_internal->at(offset);
						is_tree_edge->at(offset) = 2;
						is_tree_edge->at(reverse_index) = 2;

						if (row < column)
							(*non_tree_edges_internal)->push_back(offset);
						else
							(*non_tree_edges_internal)->push_back(
									reverse_index);

						continue;
					} else
						continue;
				}
			}

		}

		std::vector<unsigned> *run_dfs(unsigned row) {
			dfs(row);

			assert(spanning_tree->size() == Nodes - 1);

			return spanning_tree;
		}

		~DFS_HELPER() {
			visited->clear();
			parent->clear();
			is_tree_edge->clear();
		}

	};

	DFS_HELPER helper(non_tree_edges, rows, columns, rowOffsets, reverse_edge,
			Nodes);

	std::vector<unsigned> *spanning_tree = helper.run_dfs(src);

	return spanning_tree;
}
