#ifndef __CYCLES_H
#define __CYCLES_H

#include "bit_vector.h"
#include <unordered_map>
#include <assert.h>
#include <set>
#include <utility>

struct cycle
{
	csr_tree *tree;
	unsigned non_tree_edge_index;
	int total_length;

	int ID;

	bool operator<(const cycle &rhs) const
	{
		return (total_length < rhs.total_length);
	}

	struct compare
	{
		bool operator()(cycle *lhs,cycle *rhs)
		{
			return (lhs->total_length < rhs->total_length);
		}
	};

	cycle(csr_tree *tr, unsigned index)
	{
		tree = tr;
		non_tree_edge_index = index;
	}

	unsigned get_root()
	{
		return tree->root;
	}

	std::set<unsigned> *get_edges()
	{
		std::set<unsigned> *edges = new std::set<unsigned>();

		csr_multi_graph *parent_graph = tree->parent_graph;

		unsigned row = parent_graph->rows->at(non_tree_edge_index);
		unsigned col = parent_graph->columns->at(non_tree_edge_index);

		while(row != tree->root)
		{
			unsigned edge_offset = tree->parent_edges->at(row);
			unsigned reverse_edge_offset = parent_graph->reverse_edge->at(edge_offset);

			edges->insert(std::min(edge_offset,reverse_edge_offset));

			row = parent_graph->rows->at(edge_offset);
		}

		while(col != tree->root)
		{
			unsigned edge_offset = tree->parent_edges->at(col);
			unsigned reverse_edge_offset = parent_graph->reverse_edge->at(edge_offset);

			edges->insert(std::min(edge_offset,reverse_edge_offset));

			col = parent_graph->rows->at(edge_offset);
		}

		edges->insert(std::min(non_tree_edge_index,parent_graph->reverse_edge->at(non_tree_edge_index)));

		return edges;
	}



	/**
	 * @brief This method returns a bit_vector corresponding to the edges of the cycle.
	 * @details The cycles are represented using bit_vectors of non_tree edges. Non_tree edges present in 
	 * the cycles are marked set in the bit_vectors at their corresponding positions.
	 * 
	 * @param non_tree_edges map of non_tree edges and its position from 0 - non_tree_edges.size() - 1
 	 * @return bit_vector describing the cycle.
	 */
	bit_vector *get_cycle_vector(std::vector<std::pair<bool,int>> &non_tree_edges,int num_elements)
	{
		bit_vector *vector = new bit_vector(num_elements);

		unsigned row = tree->parent_graph->rows->at(non_tree_edge_index);
		unsigned col = tree->parent_graph->columns->at(non_tree_edge_index);

		std::pair<bool,int> &edge = non_tree_edges[non_tree_edge_index];

		//set flag for the current edge
		if(edge.first)
			vector->set_bit(edge.second,true);

		unsigned edge_offset;

		//check for vertices row =====> root.
		while(tree->parent_edges->at(row) != -1)
		{
			edge_offset = tree->parent_edges->at(row);

			std::pair<bool,int> &curr_edge = non_tree_edges[edge_offset];

			if(curr_edge.first)
				vector->set_bit(curr_edge.second,true);

			if(tree->parent_graph->rows->at(edge_offset) != row)
				row = tree->parent_graph->rows->at(edge_offset);
			else
				row = tree->parent_graph->columns->at(edge_offset);

			assert(row != -1);
		}

		//check for vertices col =====> root.
		while(tree->parent_edges->at(col) != -1)
		{
			edge_offset = tree->parent_edges->at(col);

			std::pair<bool,int> &curr_edge = non_tree_edges[edge_offset];
			
			if(curr_edge.first)
				vector->set_bit(curr_edge.second,true);

			if(tree->parent_graph->rows->at(edge_offset) != col)
				col = tree->parent_graph->rows->at(edge_offset);
			else
				col = tree->parent_graph->columns->at(edge_offset);

			assert(col != -1);
		}
		return vector;
	}

	void print()
	{
		printf("=================================================================================\n");
		printf("Root is %u\n",tree->root + 1);
		printf("Edge is %u - %u\n",tree->parent_graph->rows->at(non_tree_edge_index) + 1,
				 tree->parent_graph->columns->at(non_tree_edge_index) + 1);
		printf("Total weight = %d\n",total_length);
		printf("=================================================================================\n");
	}

	void print_line()
	{
		printf("{%u,(%u - %u)} ",tree->root + 1,tree->parent_graph->rows->at(non_tree_edge_index) + 1,
			tree->parent_graph->columns->at(non_tree_edge_index) + 1);
	}

};

#endif