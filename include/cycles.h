#ifndef __CYCLES_H
#define __CYCLES_H

#include "bit_vector.h"
#include <unordered_map>
#include <assert.h>

struct cycle
{
	csr_tree *tree;
	unsigned non_tree_edge_index;
	int total_length;

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

	/**
	 * @brief This method returns a bit_vector corresponding to the edges of the cycle.
	 * @details The cycles are represented using bit_vectors of non_tree edges. Non_tree edges present in 
	 * the cycles are marked set in the bit_vectors at their corresponding positions.
	 * 
	 * @param non_tree_edges map of non_tree edges and its position from 0 - non_tree_edges.size() - 1
 	 * @return bit_vector describing the cycle.
	 */
	bit_vector *get_cycle_vector(std::unordered_map<unsigned,unsigned> &non_tree_edges)
	{
		int num_elements = non_tree_edges.size();

		bit_vector *vector = new bit_vector(num_elements);

		unsigned reverse_edge = tree->parent_graph->reverse_edge->at(non_tree_edge_index);

		unsigned row = tree->parent_graph->rows->at(non_tree_edge_index);
		unsigned col = tree->parent_graph->columns->at(non_tree_edge_index);

		//set flag for the current edge

		if(non_tree_edges.find(non_tree_edge_index) != non_tree_edges.end())
			vector->set_bit(non_tree_edges[non_tree_edge_index],true);

		else if(non_tree_edges.find(reverse_edge) != non_tree_edges.end())
			vector->set_bit(non_tree_edges[reverse_edge],true);

		//check for vertices row =====> root.
		while(tree->parent_edges->at(row) != -1)
		{
			unsigned edge_offset = tree->parent_edges->at(row);
			unsigned reverse_edge_offset = tree->parent_graph->reverse_edge->at(edge_offset);

			if(non_tree_edges.find(edge_offset) != non_tree_edges.end())
				vector->set_bit(non_tree_edges[edge_offset],true);

			else if(non_tree_edges.find(reverse_edge_offset) != non_tree_edges.end())
				vector->set_bit(non_tree_edges[reverse_edge_offset],true);

			if(tree->parent_graph->rows->at(edge_offset) != row)
				row = tree->parent_graph->rows->at(edge_offset);
			else
				row = tree->parent_graph->columns->at(edge_offset);

			assert(row != -1);
		}

		//check for vertices col =====> root.
		while(tree->parent_edges->at(col) != -1)
		{
			unsigned edge_offset = tree->parent_edges->at(col);
			unsigned reverse_edge_offset = tree->parent_graph->reverse_edge->at(edge_offset);

			if(non_tree_edges.find(edge_offset) != non_tree_edges.end())
				vector->set_bit(non_tree_edges[edge_offset],true);

			else if(non_tree_edges.find(reverse_edge_offset) != non_tree_edges.end())
				vector->set_bit(non_tree_edges[reverse_edge_offset],true);

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

};

#endif