#ifndef _H_CYCLE_SEARCH
#define _H_CYCLE_SEARCH

#include <unordered_map>
#include <vector>
#include <utility>
#include <climits>

#include "cycles.h"

struct list_common_cycles
{
	std::vector<cycle*> listed_cycles;

	list_common_cycles(cycle *cle)
	{
		listed_cycles.push_back(cle);
	}

	inline void add_cycle(cycle *cle)
	{
		listed_cycles.push_back(cle);
	}

/*
 * This method is used to obtain a cycle which has the same edges but a different root.
 * The same edges are verified using the edge indexes which are stored in a set. The set is 
 * then compared to obtain the cycles. min(edges,reverse_edges) are stored in the set.
 */
	cycle *get_cycle(cycle *main_cycle,unsigned edge_offset)
	{
		if(edge_offset != UINT_MAX)
		{
			for(int i=0;i<listed_cycles.size();i++)
			{
				cycle *cle = listed_cycles[i];

				if(main_cycle->total_length != cle->total_length)
					continue;

				if(cle->non_tree_edge_index == edge_offset)
					return cle;

				else if(cle->tree->parent_graph->reverse_edge->at(edge_offset) == cle->non_tree_edge_index)
					return cle;
			}
		}
		else
		{
			std::set<unsigned> *edges_main_cycle = main_cycle->get_edges();

			//we need to search through the cycles.
			for(int i=0;i<listed_cycles.size();i++)
			{
				cycle *cle = listed_cycles[i];

				if(main_cycle->total_length != cle->total_length)
					continue;

				std::set<unsigned> *current_cycle_edges = cle->get_edges();

				if( *current_cycle_edges == *edges_main_cycle)
				{
					edges_main_cycle->clear();
					current_cycle_edges->clear();

					return cle;
				}
				else
					current_cycle_edges->clear();
			}

			edges_main_cycle->clear();
		}
		return NULL;
	}
};

struct cycle_storage
{
	int Nodes;
	std::vector<std::unordered_map<unsigned long long,list_common_cycles*> > list_cycles;
	std::vector<csr_tree*> list_trees;

	inline unsigned long long combine(unsigned u,unsigned v)
	{
		unsigned long long value = u;
		value <<= 32;

		value = value | v;

		return value;
	}

	cycle_storage(int N)
	{
		Nodes = N;
		list_cycles.resize(Nodes);
		list_trees.resize(Nodes);
		for(int i=0;i<Nodes;i++)
			list_trees[i] = NULL;
	}

	~cycle_storage()
	{
		list_cycles.clear();
	}

	void add_cycle(unsigned root,unsigned u,unsigned v,cycle *cle)
	{
		unsigned long long index = combine(std::min(u,v),std::max(u,v));

		if(list_cycles[root].find(index) == list_cycles[root].end()) 
			list_cycles[root].insert(std::make_pair(index,new list_common_cycles(cle)));
		else
			list_cycles[root][index]->add_cycle(cle);
	}

	cycle *get_cycle(unsigned root,unsigned u,unsigned v,cycle *main_cycle,unsigned edge_offset = UINT_MAX)
	{
		unsigned long long index = combine(std::min(u,v),std::max(u,v));

		if(list_cycles[root].find(index) == list_cycles[root].end())
		{
			return NULL;
		}
		else
		{
			cycle *cle = list_cycles[root][index]->get_cycle(main_cycle,edge_offset);
			return cle;
		}

	}

	std::vector<unsigned>* get_s_value(unsigned root)
	{
		if(list_trees[root] == NULL)
			return NULL;
		return list_trees[root]->s_values;
	}

	void add_trees(std::vector<csr_tree*> &list_of_trees)
	{
		for(int i=0;i<list_of_trees.size();i++)
		{
			int root = list_of_trees[i]->root;
			list_trees[root] = list_of_trees[i];
		}
	}

	void clear_cycles()
	{
		for(int i=0;i<list_cycles.size();i++)
		{
			list_cycles[i].clear();
		}
		list_cycles.clear();
	}
};

#endif