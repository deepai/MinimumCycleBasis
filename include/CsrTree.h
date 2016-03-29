#ifndef __CSR_TREE_H
#define __CSR_TREE_H

#include <assert.h>
#include "CsrGraph.h"

class csr_tree
{
public:
	csr_graph *parent_graph;
	std::vector<unsigned> *tree_edges;
	std::vector<unsigned> *non_tree_edges = NULL;

	csr_tree(csr_graph *graph)
	{
		parent_graph = graph;
		assert (parent_graph != NULL);
	}

	void populate_tree_edges(bool populate_non_tree_edges)
	{
		if(populate_non_tree_edges)
			non_tree_edges = new std::vector<unsigned>();

		tree_edges = parent_graph->get_spanning_tree(&non_tree_edges);
	}

	inline void get_edge_endpoints(unsigned &row,unsigned &col,unsigned &offset)
	{
		parent_graph->get_edge_endpoints(row,col,offset);
	}
};

#endif 