#ifndef __CSR_MULTI_GRAPH
#define __CSR_MULTI_GRAPH

#include "CsrGraph.h"

class csr_multi_graph : public csr_graph{

protected:
	struct edge
	{
		unsigned row;
		unsigned col;
		int weight;

		struct edge *reverse_edge_ptr;
		int reverse_edge_index;

		edge(unsigned &r,unsigned &c,int &w)
		{
			row = r;
			col = c;
			weight = w;
		}
	};

	struct compare { 
		bool operator()(const edge *a, const edge *b) const {
			if(a->row == b->row)
				return (a->col < b->col);
			else
				return (a->row < b->row);
		}
	};
public:
	std::vector<unsigned> *reverse_edge;
	csr_multi_graph()
	{
		reverse_edge = new std::vector<unsigned>();
	}

	void insert(int a,int b,int wt,bool direction)
	{
		columns->push_back(b);
		rows->push_back(a);
		weights->push_back(wt);

		if(!direction)
			reverse_edge->push_back(rows->size());
		else
			reverse_edge->push_back(rows->size()-2);

		if(!direction)
			insert(b,a,wt,true);
	}
	//Calculate the degree of the vertices and create the rowOffset
	void calculateDegreeandRowOffset()
	{
		rowOffsets->resize(Nodes + 1);
		degree->resize(Nodes);

		for(int i=0;i<Nodes;i++)
		{
			rowOffsets->at(i) = 0;
			degree->at(i) = 0;
		}

		rowOffsets->at(Nodes) = 0;

		//Allocate a pair array for rows and columns array
		std::vector<edge*> combined;
		
		//copy the elements from the row and column array
		for(int i=0;i<rows->size();i++)
			combined.push_back(new edge(rows->at(i),columns->at(i),weights->at(i)));

		//assing the reverse_edge_pointers to the correct edge pointers.
		for(int i=0;i<rows->size();i++)
			combined[i]->reverse_edge_ptr = combined[reverse_edge->at(i)];

		//Sort the elements first by row, then by column
		std::sort(combined.begin(),combined.end(),compare());

		for(int i=0;i<rows->size();i++)
			combined[i]->reverse_edge_ptr->reverse_edge_index = i;

		//copy back the elements into row and columns
		for(int i=0;i<rows->size();i++)
		{
			rows->at(i) = combined[i]->row;
			columns->at(i) = combined[i]->col;
			weights->at(i) = combined[i]->weight;

			assert(combined[i]->reverse_edge_index < rows->size());

			reverse_edge->at(i) = combined[i]->reverse_edge_index;
		}

		for(int i=0;i<rows->size();i++)
			delete combined[i];

		combined.clear();

		//Now calculate the row_offset

		for(int i=0;i<rows->size();i++)
		{
			unsigned curr_row = rows->at(i);
			
			rowOffsets->at(curr_row)++;
		}

		unsigned prev = 0,current;

		for(int i=0;i<=Nodes;i++)
		{
			current = rowOffsets->at(i);
			rowOffsets->at(i) = prev;
			prev += current;
		}

		for(int i=0;i<Nodes;i++)
		{
			degree->at(i) = rowOffsets->at(i+1) - rowOffsets->at(i);
		}

		assert(rowOffsets->at(Nodes) == rows->size());

		
		#ifdef INFO
			printf("row_offset size = %d,columns size = %d\n",rowOffsets->size(),columns->size());
		#endif

	}

	std::vector<unsigned> *csr_multi_graph::get_spanning_tree(std::vector<unsigned> **non_tree_edges,
						    int src);

	static csr_multi_graph *get_modified_graph(csr_graph *graph,
				std::vector<unsigned> *remove_edge_list,
				std::vector<std::vector<unsigned> > *edges_new_list,
				int nodes_removed)
	{
		std::vector<bool> filter_edges(graph->rows->size());
		for(int i=0;i<filter_edges.size();i++)
			filter_edges[i] = false;

		for(int i=0;(remove_edge_list!=NULL) &&  (i<remove_edge_list->size());i++)
			filter_edges[remove_edge_list->at(i)] = true;

		csr_multi_graph *new_reduced_graph = new csr_multi_graph();

		new_reduced_graph->Nodes = graph->Nodes - nodes_removed;

		//add new edges first.
		for(int i=0;(edges_new_list != NULL) && (i<edges_new_list->size());i++)
		{
			new_reduced_graph->insert(edges_new_list->at(i)[0],
						  edges_new_list->at(i)[1],
						  edges_new_list->at(i)[2],
						  false);
		}

		for(int i=0;i<graph->rows->size();i++)
		{
			if(!filter_edges.at(i))
			{
				if(graph->rows->at(i) < graph->columns->at(i))
					new_reduced_graph->insert(graph->rows->at(i),
								  graph->columns->at(i),
								  graph->weights->at(i),
								  false);
			}
		}

		new_reduced_graph->calculateDegreeandRowOffset();

		filter_edges.clear();

		return new_reduced_graph;
	}
};

#endif