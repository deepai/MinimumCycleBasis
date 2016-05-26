#ifndef __CSR_TREE_H
#define __CSR_TREE_H

#include <assert.h>
#include "CsrGraphMulti.h"
#include "Dijkstra.h"

class csr_tree {
public:
	int root;
	csr_multi_graph *parent_graph;
	std::vector<unsigned> *tree_edges;
	std::vector<unsigned> *non_tree_edges = NULL;

	std::vector<int> *parent_edges;
	std::vector<int> *distance;

	struct compare {
		csr_multi_graph *parent_graph;
		compare(csr_multi_graph *graph) {
			parent_graph = graph;
		}
		bool operator()(const unsigned &A, const unsigned &B) const {
			return (parent_graph->weights->at(A) < parent_graph->weights->at(B));
		}
	};

	csr_tree(csr_multi_graph *graph) {
		parent_graph = graph;
		assert(parent_graph != NULL);
		assert(parent_graph->rows->size() == parent_graph->columns->size());
		assert(parent_graph->rowOffsets->size() == parent_graph->Nodes + 1);
	}

	~csr_tree() {
		tree_edges->clear();
		non_tree_edges->clear();
		parent_edges->clear();
		distance->clear();

		tree_edges = NULL;
		non_tree_edges = NULL;
		parent_edges = NULL;
		distance = NULL;

	}

	void populate_tree_edges(bool populate_non_tree_edges, int &src) {
		if (populate_non_tree_edges)
			non_tree_edges = new std::vector<unsigned>();

		root = src;

		tree_edges = parent_graph->get_spanning_tree(&non_tree_edges, src);

		//std::sort(non_tree_edges->begin(),non_tree_edges->end(),compare(parent_graph));
	}

	void obtain_shortest_path_tree(dijkstra &helper,
			bool populate_non_tree_edges, int src) {
		if (populate_non_tree_edges)
			non_tree_edges = new std::vector<unsigned>();

		root = src;

		helper.dijkstra_sp(src);
		helper.compute_non_tree_edges(&non_tree_edges);

		tree_edges = helper.tree_edges;
		parent_edges = new std::vector<int>();

		for (int i = 0; i < helper.edge_offsets.size(); i++)
			parent_edges->push_back(helper.edge_offsets[i]);

		distance = new std::vector<int>();

		for (int i = 0; i < helper.distance.size(); i++)
			distance->push_back(helper.distance[i]);
	}

	inline void get_edge_endpoints(unsigned &row, unsigned &col, int &weight,
			unsigned &offset) {
		parent_graph->get_edge_endpoints(row, col, weight, offset);
	}

	std::vector<int> *get_parent_edges() {
		std::vector<int> *v = new std::vector<int>();

		assert(tree_edges != NULL);

		unsigned row, col;

		v->at(root) = -1;

		for (int i = 0; i < tree_edges->size(); i++) {
			unsigned edge = tree_edges->at(i);
			row = parent_graph->rows->at(i);
			col = parent_graph->columns->at(i);

			v->at(col) = row;
		}
		return v;
	}

	void print_tree_edges() {
		printf(
				"=================================================================================\n");
		printf("Printing Spanning Tree Edges,count = %d\n", tree_edges->size());
		for (int i = 0; i < tree_edges->size(); i++)
			printf("%u %u %d\n", parent_graph->rows->at(tree_edges->at(i)) + 1,
					parent_graph->columns->at(tree_edges->at(i)) + 1,
					parent_graph->weights->at(tree_edges->at(i)));
		printf(
				"=================================================================================\n");
	}

	void print_non_tree_edges() {
		if (non_tree_edges == NULL)
			return;

		printf(
				"=================================================================================\n");
		printf("Printing Non-Tree Edges,count = %d\n", non_tree_edges->size());
		for (int i = 0; i < non_tree_edges->size(); i++)
			printf("%u %u\n", parent_graph->rows->at(non_tree_edges->at(i)) + 1,
					parent_graph->columns->at(non_tree_edges->at(i)) + 1);
		printf(
				"=================================================================================\n");
	}
};

#endif 
