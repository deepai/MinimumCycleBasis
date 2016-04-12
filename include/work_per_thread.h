#ifndef _H_WORK_PER_THREAD
#define _H_WORK_PER_THREAD

#include <vector>
#include "CsrTree.h"
#include "cycles.h"
#include "Dijkstra.h"
#include "bit_vector.h"

#include <unordered_map>


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
				cle->total_length = total_weight;
				list_cycles.push_back(cle);
			}
		}

		sp_tree->remove_non_tree_edges();
		sp_tree->node_pre_compute = new std::vector<unsigned>(graph->Nodes);
	}

	void empty_cycles()
	{
		list_cycles.clear();
	}

	void precompute_supportVec(std::unordered_map<unsigned,unsigned> &non_tree_edge_map,bit_vector &vector)
	{
		for(int i=0;i<shortest_path_trees.size();i++)
		{
			csr_tree *current_tree = shortest_path_trees.at(i);

			std::vector<unsigned> *tree_edges = current_tree->tree_edges;
			std::vector<unsigned> *precompute_nodes = current_tree->node_pre_compute;
			csr_multi_graph *graph = current_tree->parent_graph;

			precompute_nodes->at(current_tree->root) = 0;

			unsigned edge_offset,reverse_edge,row,column,position,bit;

			for(int i=0;i<tree_edges->size();i++)
			{
				edge_offset = tree_edges->at(i);
				reverse_edge = graph->reverse_edge->at(edge_offset);
				row = graph->rows->at(edge_offset);
				column = graph->columns->at(edge_offset);

				//non_tree_edge
				if(non_tree_edge_map.find(edge_offset) != non_tree_edge_map.end())
				{
					bit = vector.get_bit(non_tree_edge_map.at(edge_offset));
					precompute_nodes->at(column) = (precompute_nodes->at(row) + bit)%2;
				}
				else if(non_tree_edge_map.find(reverse_edge) != non_tree_edge_map.end())
				{
					bit = vector.get_bit(non_tree_edge_map.at(reverse_edge));
					precompute_nodes->at(column) = (precompute_nodes->at(row) + bit)%2;
				}
				else //tree edge
					precompute_nodes->at(column) = precompute_nodes->at(row);
			}
		}
	}
};

#endif