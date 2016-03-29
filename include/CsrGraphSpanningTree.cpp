#include <queue>
#include "CsrGraph.h"

/**
 * @brief This method is used to obtain the spanning tree of a graph. The spanning tree contains edge offsets from the csr_graph
 * @details 
 *  
 * @param  address of an vector for storing non_tree_edges;
 * @return vector of edge_offsets in bfs ordering.
 */
std::vector<unsigned> *csr_graph::get_spanning_tree(std::vector<unsigned> **non_tree_edges)
{
	std::vector<unsigned> *spanning_tree = 
		new std::vector<unsigned>();

	std::vector<bool> *visited = new std::vector<bool>(Nodes);
	for(int i=0;i<Nodes;i++)
		visited->at(i) = false;

	std::queue<unsigned> bfs_queue;

	bfs_queue.push(rows->at(0));

	visited->at(rows->at(0)) = true;

	while(!bfs_queue.empty())
	{
		unsigned row = bfs_queue.front();
		bfs_queue.pop();

		for(unsigned offset = rowOffsets->at(row); offset < rowOffsets->at(row + 1); offset++)
		{
			unsigned column = columns->at(offset);
			if(!visited->at(column))
			{
				visited->at(column) = true;
				spanning_tree->push_back(offset);
			}
			else
			{
				if((row < column) && (*non_tree_edges != NULL))
					(*non_tree_edges)->push_back(offset);
			}
		}
	}

	visited->clear();

	assert (spanning_tree->size() == Nodes - 1);
	assert ((*non_tree_edges == NULL) || ((*non_tree_edges)->size() == (rows->size()/2 - Nodes + 1)));

	return spanning_tree;

}