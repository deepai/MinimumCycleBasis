#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <utility>
#include <set>
#include <stack>
#include <omp.h>
#include <string>
#include <algorithm>
#include <atomic>
#include <list>
#include <unordered_set>
#include <cstring>
#include <utility>

#include "FileReader.h"
#include "Files.h"
#include "utils.h"
#include "Host_Timer.h"
#include "CsrGraph.h"
#include "CsrTree.h"
#include "bit_vector.h"
#include "work_per_thread.h"
#include "cycle_searcher.h"
#include "compressed_trees.h"
#include "FVS.h"

debugger dbg;
HostTimer globalTimer;

std::string InputFileName;
std::string OutputFileDirectory;

double totalTime = 0;

int main(int argc,char* argv[])
{
	if(argc < 4)
	{
		printf("Ist Argument should indicate the InputFile\n");
		printf("2nd Argument should indicate the outputdirectory\n");
		printf("3th argument should indicate the number of threads.(Optional) (1 default)\n");
		exit(1);
	}

	int num_threads = 1;

	if(argc == 4)
		num_threads = atoi(argv[3]);

	InputFileName = argv[1];

    	omp_set_num_threads(num_threads);

    	//Open the FileReader class
	std::string InputFilePath = InputFileName;

	//Read the Inputfile.
	FileReader Reader(InputFilePath.c_str());

	int v1,v2,Initial_Vertices,weight;;

	int nodes,edges,chunk_size = 1;

	//firt line of the input file contains the number of nodes and edges
	Reader.get_nodes_edges(nodes,edges); 

	csr_graph *graph=new csr_graph();

	graph->Nodes = nodes;
	/*
	 * ====================================================================================
	 * Fill Edges.
	 * ====================================================================================
	 */
	for(int i=0;i<edges;i++)
	{
		Reader.read_edge(v1,v2,weight);
		graph->insert(v1,v2,weight,false);
	}

	graph->calculateDegreeandRowOffset();

	Reader.fileClose();

	if(graph->get_num_degree_two_vertices() == graph->Nodes)
	{
		printf("Graph is a cycle\n");		
		return 0;
	}

	int source_vertex = 0;

	csr_multi_graph *reduced_graph = csr_multi_graph::get_modified_graph(graph,
									     NULL,
									     NULL,
									     0);

	//Node Validity
	assert(reduced_graph->Nodes + 0 == graph->Nodes);

	reduced_graph->print_graph();

	FVS fvs_helper(reduced_graph);

	fvs_helper.MGA();

	fvs_helper.print_fvs();

	int *fvs_array = fvs_helper.get_copy_fvs_array();

	csr_tree *initial_spanning_tree = new csr_tree(reduced_graph);
	initial_spanning_tree->populate_tree_edges(true,source_vertex);

	int num_non_tree_edges = initial_spanning_tree->non_tree_edges->size();

	//Spanning Tree Validity
	assert(num_non_tree_edges == reduced_graph->rows->size()/2 - reduced_graph->Nodes + 1);

	initial_spanning_tree->print_tree_edges();
	initial_spanning_tree->print_non_tree_edges();

	std::vector<int> non_tree_edges_map(reduced_graph->rows->size());
	std::fill(non_tree_edges_map.begin(),non_tree_edges_map.end(),-1);
	
	debug("Map of non-tree edges");

	for(int i=0;i<initial_spanning_tree->non_tree_edges->size();i++)
	{
		non_tree_edges_map[initial_spanning_tree->non_tree_edges->at(i)] = i;

		printf("%d : %u - %u\n",i,reduced_graph->rows->at(initial_spanning_tree->non_tree_edges->at(i)) + 1,
			reduced_graph->columns->at(initial_spanning_tree->non_tree_edges->at(i)) + 1);
	}

	for(int i=0;i<reduced_graph->rows->size();i++)
	{
		//copy the edges into the reverse edges as well.
		if(non_tree_edges_map[i] < 0)
			if(non_tree_edges_map[reduced_graph->reverse_edge->at(i)] >=0 )
				non_tree_edges_map[i] = non_tree_edges_map[reduced_graph->reverse_edge->at(i)];
	}

	//construct the initial
	compressed_trees trees(chunk_size,fvs_helper.get_num_elements(),fvs_array,reduced_graph);

	cycle_storage *storage = new cycle_storage(reduced_graph->Nodes);

	worker_thread **multi_work = new worker_thread*[num_threads];

	for(int i=0;i<num_threads;i++)
		multi_work[i] = new worker_thread(reduced_graph,storage,fvs_array,&trees);

	int count_cycles = 0;

	//produce shortest path trees across all the nodes.
	#pragma omp parallel for reduction(+:count_cycles)
	for(int i = 0; i < trees.fvs_size; ++i)
	{
		int threadId = omp_get_thread_num();

		count_cycles += multi_work[threadId]->produce_sp_tree_and_cycles(i,reduced_graph);
	}

	std::vector<cycle*> list_cycle_vec;
	std::list<cycle*> list_cycle;

	for(int j=0;j<storage->list_cycles.size();j++)
	{
		for(std::unordered_map<unsigned long long,list_common_cycles*>::iterator it = storage->list_cycles[j].begin();
			it != storage->list_cycles[j].end(); it++)
		{
			for(int k=0;k<it->second->listed_cycles.size();k++)
			{
				list_cycle_vec.push_back(it->second->listed_cycles[k]);
				list_cycle_vec.back()->ID = list_cycle_vec.size() - 1;
			}
		}
	}

	sort(list_cycle_vec.begin(),list_cycle_vec.end(),cycle::compare());

	printf("\nList Cycles Pre Isometric\n");
	for(std::vector<cycle*>::iterator cycle = list_cycle_vec.begin();
			cycle != list_cycle_vec.end(); cycle++)
	{
		printf("%u-(%u - %u) : %d\n",((*cycle))->get_root() + 1,
					reduced_graph->rows->at((*cycle)->non_tree_edge_index) + 1,
					reduced_graph->columns->at((*cycle)->non_tree_edge_index) + 1,
					(*cycle)->total_length);
	}

	printf("\n\n");

	//isometric_cycle *isometric_cycle_helper = new isometric_cycle(list_cycle_vec.size(),storage,&list_cycle_vec);

	//isometric_cycle_helper->obtain_isometric_cycles();

	//delete isometric_cycle_helper;


	for(int i=0; i<list_cycle_vec.size(); i++)
	{
		if(list_cycle_vec[i] != NULL)
			list_cycle.push_back(list_cycle_vec[i]);
	}

	list_cycle_vec.clear();

	printf("\nList Cycles Post Isometric\n");
	for(std::list<cycle*>::iterator cycle = list_cycle.begin();
			cycle != list_cycle.end(); cycle++)
	{
		printf("%u-(%u - %u) : %d\n",((*cycle))->get_root() + 1,
					reduced_graph->rows->at((*cycle)->non_tree_edge_index) + 1,
					reduced_graph->columns->at((*cycle)->non_tree_edge_index) + 1,
					(*cycle)->total_length);
	}
	printf("\n");

	//At this stage we have the shortest path trees and the cycles sorted in increasing order of length.

	//generate the bit vectors
	bit_vector **support_vectors = new bit_vector*[num_non_tree_edges];

	printf("Number of non_tree_edges = %d\n",num_non_tree_edges);

	for(int i=0;i<num_non_tree_edges;i++)
	{
		support_vectors[i] = new bit_vector(num_non_tree_edges);
		support_vectors[i]->set_bit(i,true);
	}

	std::vector<cycle*> final_mcb;

	//Main Outer Loop of the Algorithm.
	for(int e=0;e<num_non_tree_edges;e++)
	{
		debug("Si is as follows.",e);
		support_vectors[e]->print();
		#pragma omp parallel for
		for(int i=0;i<num_threads;i++)
		{
			multi_work[i]->precompute_supportVec(non_tree_edges_map,*support_vectors[e]);
		}

		unsigned *node_rowoffsets,*node_columns,*precompute_nodes;
		int *node_edgeoffsets,*node_parents,*node_distance;
		unsigned src,edge_offset,reverse_edge,row,col,position,bit;
		int src_index;

		for(std::list<cycle*>::iterator cycle = list_cycle.begin();
			cycle != list_cycle.end(); cycle++)
		{
			src = (*cycle)->get_root();
			src_index = trees.vertices_map[src];

			trees.get_node_arrays(&node_rowoffsets,&node_columns,&node_edgeoffsets,&node_parents,&node_distance,src_index);
			trees.get_precompute_array(&precompute_nodes,src_index);

			edge_offset = (*cycle)->non_tree_edge_index;
			bit = 0;

			unsigned row,col;
			row = reduced_graph->rows->at(edge_offset);
			col = reduced_graph->columns->at(edge_offset);

			if(non_tree_edges_map[edge_offset] >= 0)
			{
				bit = support_vectors[e]->get_bit(non_tree_edges_map[edge_offset]);
			}

			bit = (bit + precompute_nodes[row])%2;
			bit = (bit + precompute_nodes[col])%2;

			if(bit == 1)
			{

				final_mcb.push_back(*cycle);
				list_cycle.erase(cycle);
				break;
			}
		}

		bit_vector *cycle_vector = final_mcb.back()->get_cycle_vector(non_tree_edges_map,
																	  initial_spanning_tree->non_tree_edges->size());
		final_mcb.back()->print();

		printf("Ci ");
		cycle_vector->print();

		#pragma omp parallel for 
		for(int j=e+1;j<num_non_tree_edges;j++)
		{
			unsigned product = cycle_vector->dot_product(support_vectors[j]);
			if(product == 1)
				support_vectors[j]->do_xor(support_vectors[e]);
			printf("%d ",product);
			support_vectors[j]->print();
		}
	}

	list_cycle.clear();

	debug("\nPrinting final mcbs\n");

	int total_weight = 0;

	for(int i=0;i<final_mcb.size();i++)
	{
		final_mcb[i]->print();

		total_weight += final_mcb[i]->total_length;
	}

	printf("Total Weight = %d\n",total_weight);

	delete[] fvs_array;

	return 0;
}