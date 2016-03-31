#ifndef __CSR_MULTI_GRAPH
#define __CSR_MULTI_GRAPH

#include "CsrGraph.h"

class csr_multi_graph : public csr_graph{
public:
	std::vector<unsigned> *reverse_edge;
	csr_multi_graph()
	{
		reverse_edge = new std::vector<unsigned>();
	}

	csr_multi_graph *get_modified_graph(std::vector<unsigned> *remove_edge_list,
		std::vector<std::vector<unsigned> > *edges_new_list,
		int nodes_removed)
	{
		std::vector<bool> filter_edges(rows->size());
		for(int i=0;i<filter_edges.size();i++)
			filter_edges[i] = false;

		for(int i=0;i<remove_edge_list->size();i++)
			filter_edges[remove_edge_list->at(i)] = true;

		csr_multi_graph *new_reduced_graph = new csr_graph();

		new_reduced_graph->Nodes = Nodes - nodes_removed;

		//add new edges first.
		for(int i=0;i<edges_new_list->size();i++)
		{
			new_reduced_graph->insert(edges_new_list->at(i)[0],
						  edges_new_list->at(i)[1],
						  edges_new_list->at(i)[2],
						  false);
		}

		for(int i=0;i<rows->size();i++)
		{
			if(!filter_edges.at(i))
				new_reduced_graph->insert(rows->at(i),columns->at(i),weights->at(i),true);
		}

		new_reduced_graph->calculateDegreeandRowOffset();

		filter_edges.clear();

		return new_reduced_graph;
	}
};

#endif