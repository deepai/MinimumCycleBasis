#include <stack>
#include "CsrGraph.h"

/**
 * @brief This method is used to obtain the spanning tree of a graph. The spanning tree contains edge offsets from the csr_graph
 * @details This method may also return the list of non_tree_edges and ear decomposition corresponding to every non-tree 
 * 	    edges.
 *  
 * @param  address of an vector for storing non_tree_edges,ear decomposition vector;
 * @return vector of edge_offsets in bfs ordering.
 */
std::vector<unsigned> *csr_graph::get_spanning_tree(
		std::vector<unsigned> **non_tree_edges,
		std::vector<unsigned> *ear_decomposition, int src) {
	struct DFS_HELPER {
		int Nodes;

		std::vector<unsigned> *spanning_tree;
		std::vector<bool> *visited;
		std::vector<unsigned> *ear_decomposition_internal;

		std::vector<unsigned> *rows_internal;
		std::vector<unsigned> *columns_internal;
		std::vector<unsigned> *rowOffsets_internal;

		std::vector<unsigned> **non_tree_edges_internal;
		std::vector<unsigned> *stack;

		std::vector<int> *parent;

		int ear_count;

		DFS_HELPER(std::vector<unsigned> **non_tree_edges,
				std::vector<unsigned> *rows, std::vector<unsigned> *columns,
				std::vector<unsigned> *rowOffsets,
				std::vector<unsigned> *ear_decomposition, int _nodes) {
			spanning_tree = new std::vector<unsigned>();
			visited = new std::vector<bool>();
			stack = new std::vector<unsigned>();
			parent = new std::vector<int>();

			Nodes = _nodes;
			ear_count = 0;

			for (int i = 0; i < Nodes; i++) {
				parent->push_back(-1);
				visited->push_back(false);
			}

			non_tree_edges_internal = non_tree_edges;

			rows_internal = rows;
			columns_internal = columns;
			rowOffsets_internal = rowOffsets;

			ear_decomposition_internal = ear_decomposition;

			assert((ear_decomposition_internal->size() == Nodes + 1));
		}

		void dfs(unsigned row) {
			visited->at(row) = true;

			stack->push_back(row);

			for (unsigned offset = rowOffsets_internal->at(row);
					offset < rowOffsets_internal->at(row + 1); offset++) {
				unsigned column = columns_internal->at(offset);
				if (!visited->at(column)) {
					visited->at(column) = true;
					spanning_tree->push_back(offset);
					parent->at(column) = row;
					dfs(column);
				} else {
					bool ear_incremented = false;

					if (column == parent->at(row))
						continue;

					(*non_tree_edges_internal)->push_back(offset);

					if (ear_decomposition_internal != NULL) {
						for (std::vector<unsigned>::reverse_iterator it =
								stack->rbegin(); it != stack->rend(); it++) {
							if (ear_decomposition_internal->at(*it) == 0) {
								ear_decomposition_internal->at(*it) = ear_count
										+ 1;
								ear_incremented = true;
							} else
								break;
						}
					}
					if (ear_incremented)
						ear_count++;
				}
			}

			stack->pop_back();
		}

		std::vector<unsigned> *run_dfs(unsigned row) {
			dfs(row);

			assert(spanning_tree->size() == Nodes - 1);

			if (ear_decomposition_internal != NULL)
				ear_decomposition_internal->at(Nodes) = ear_count;

			return spanning_tree;
		}

		~DFS_HELPER() {
			visited->clear();
			stack->clear();
			parent->clear();
		}

	};

	DFS_HELPER helper(non_tree_edges, rows, columns, rowOffsets,
			ear_decomposition, Nodes);

	std::vector<unsigned> *spanning_tree = helper.run_dfs(src);

	return spanning_tree;
}
