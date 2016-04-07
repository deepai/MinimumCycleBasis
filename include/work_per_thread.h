#ifndef _H_WORK_PER_THREAD
#define _H_WORK_PER_THREAD

#include <vector>
#include "CsrTree.h"
#include "cycles.h"
#include "Dijkstra.h"

struct worker_thread
{
	std::vector<csr_tree*> shortest_path_trees;
	std::vector<cycle*> list_cycles;
	dijkstra *helper;

	worker_thread(csr_multi_graph *graph)
	{
		helper = new dijkstra(graph->Nodes,graph);
	}


	void produce_sp_tree_and_cycles(int src,csr_multi_graph *graph)
	{
		helper->reset();

		csr_tree *sp_tree = new csr_tree(graph);

		//compute shortest path spanning tree and also non-tree edges
		sp_tree->obtain_shortest_path_tree(*helper,true,src);
		shortest_path_trees.push_back(sp_tree);

		//compute the cycles;
		std::vector<unsigned> *non_tree_edges = sp_tree->non_tree_edges;

		int total_weight;
		bool is_edge_cycle;

		for(int i=0;i<non_tree_edges->size();i++)
		{
			is_edge_cycle = helper->is_edge_cycle(non_tree_edges->at(i),total_weight,src);

			if(is_edge_cycle)
			{
				cycle *cle = new cycle(sp_tree,non_tree_edges->at(i));
				list_cycles.push_back(cle);
			}
		}

		sp_tree->remove_non_tree_edges();
	}

	void empty_cycles()
	{
		list_cycles.clear();
	}
};

#endif