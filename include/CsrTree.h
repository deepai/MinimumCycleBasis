#ifndef __CSR_TREE_H
#define __CSR_TREE_H

#include <assert.h>
#include "CsrGraphMulti.h"
#include "Dijkstra.h"

class csr_tree
{
public:
	int root;
	csr_multi_graph *parent_graph;
	std::vector<unsigned> *tree_edges;
	std::vector<unsigned> *non_tree_edges = NULL;
	std::vector<unsigned> *node_pre_compute;

	csr_tree(csr_multi_graph *graph)
	{
		parent_graph = graph;
		assert (parent_graph != NULL);
		assert (parent_graph->rows->size() == parent_graph->columns->size());
		assert (parent_graph->rowOffsets->size() == parent_graph->Nodes + 1);
	}

	void populate_tree_edges(bool populate_non_tree_edges,int &src)
	{
		if(populate_non_tree_edges)
			non_tree_edges = new std::vector<unsigned>();

		root = src;

		tree_edges = parent_graph->get_spanning_tree(&non_tree_edges,src);

		std::vector<unsigned> *reduced_non_tree_edge = new std::vector<unsigned>();

		for(int i=0;i<non_tree_edges->size();i++)
		{
			unsigned edge_offset = non_tree_edges->at(i);

			if(parent_graph->rows->at(edge_offset) < parent_graph->columns->at(edge_offset))
				reduced_non_tree_edge->push_back(edge_offset);
		}

		non_tree_edges->clear();

		non_tree_edges = reduced_non_tree_edge;
	}

	void obtain_shortest_path_tree(dijkstra &helper,bool populate_non_tree_edges,int src)
	{
		if(populate_non_tree_edges)
			non_tree_edges = new std::vector<unsigned>();

		root = src;

		helper.dijkstra_sp(src);
		helper.compute_non_tree_edges(&non_tree_edges);

		tree_edges = helper.tree_edges;
	}

	void remove_non_tree_edges()
	{
		non_tree_edges->clear();
		non_tree_edges = NULL;
	}


	inline void get_edge_endpoints(unsigned &row,unsigned &col,int &weight,unsigned &offset)
	{
		parent_graph->get_edge_endpoints(row,col,weight,offset);
	}

	std::vector<int> *get_parent_edges()
	{
		std::vector<int> *v = new std::vector<int>();

		assert(tree_edges != NULL);

		unsigned row,col;

		v->at(root) = -1;

		for(int i=0;i<tree_edges->size();i++)
		{
			unsigned edge = tree_edges->at(i);
			row = parent_graph->rows->at(i);
			col = parent_graph->columns->at(i);

			v->at(col) = row;
		}
		return v;
	}

	void print_tree_edges()
	{
		printf("=================================================================================\n");
		printf("Printing Spanning Tree Edges,count = %d\n",tree_edges->size());
		for(int i=0;i<tree_edges->size();i++)
			printf("%u %u\n",parent_graph->rows->at(tree_edges->at(i)) + 1,
					 parent_graph->columns->at(tree_edges->at(i)) + 1);
		printf("=================================================================================\n");
	}

	void print_non_tree_edges()
	{
		printf("=================================================================================\n");
		printf("Printing Non-Tree Edges,count = %d\n",non_tree_edges->size());
		for(int i=0;i<non_tree_edges->size();i++)
			printf("%u %u\n",parent_graph->rows->at(non_tree_edges->at(i)) + 1,
					 parent_graph->columns->at(non_tree_edges->at(i)) + 1);
		printf("=================================================================================\n");
	}
};

#endif 