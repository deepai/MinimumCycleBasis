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

#include "FileReader.h"
#include "Files.h"
#include "utils.h"
#include "Host_Timer.h"
#include "CsrGraph.h"
#include "CsrTree.h"
#include "bit_vector.h"
#include "work_per_thread.h"

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

	int nodes,edges;

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

	std::vector<std::vector<unsigned> > *chains = new std::vector<std::vector<unsigned> >();

	debug("Input File Reading Complete...\n");

	int source_vertex;

	std::vector<unsigned> *remove_edge_list = graph->mark_degree_two_chains(&chains,source_vertex);
	//initial_spanning_tree.populate_tree_edges(true,NULL,souce_vertex);

	std::vector<std::vector<unsigned> > *edges_new_list = new std::vector<std::vector<unsigned> >();

	int nodes_removed = 0;

	for(int i=0;i<chains->size();i++)
	{
		unsigned row,col;
		unsigned total_weight = graph->sum_edge_weights(chains->at(i),row,col);

		nodes_removed += chains->at(i).size() - 1;

		std::vector<unsigned> new_edge = std::vector<unsigned>();
		new_edge.push_back(row);
		new_edge.push_back(col);
		new_edge.push_back(total_weight);

		edges_new_list->push_back(new_edge);
		//debug(row+1,col+1,total_weight);
	}

	debug ("Number of nodes removed = ",nodes_removed);

	csr_multi_graph *reduced_graph = csr_multi_graph::get_modified_graph(graph,
									     remove_edge_list,
									     edges_new_list,
									     nodes_removed);

	//Node Validity
	assert(reduced_graph->Nodes + nodes_removed == graph->Nodes);

	reduced_graph->print_graph();

	csr_tree *initial_spanning_tree = new csr_tree(reduced_graph);
	initial_spanning_tree->populate_tree_edges(true,source_vertex);

	int num_non_tree_edges = initial_spanning_tree->non_tree_edges->size();

	//Spanning Tree Validity
	assert(num_non_tree_edges == reduced_graph->rows->size()/2 - reduced_graph->Nodes + 1);

	initial_spanning_tree->print_tree_edges();
	initial_spanning_tree->print_non_tree_edges();

	std::unordered_map<unsigned,unsigned> *non_tree_edges_map = new std::unordered_map<unsigned,unsigned>();
	
	debug("Map of non-tree edges");

	for(int i=0;i<initial_spanning_tree->non_tree_edges->size();i++)
	{
		non_tree_edges_map->insert(std::make_pair(initial_spanning_tree->non_tree_edges->at(i),i));
		printf("%d : %u - %u\n",i,reduced_graph->rows->at(initial_spanning_tree->non_tree_edges->at(i)) + 1,
			reduced_graph->columns->at(initial_spanning_tree->non_tree_edges->at(i)) + 1);
	}


	worker_thread **multi_work = new worker_thread*[num_threads];
	for(int i=0;i<num_threads;i++)
		multi_work[i] = new worker_thread(reduced_graph);

	int count_cycles = 0;

	//produce shortest path trees across all the nodes.
	#pragma omp parallel for reduction(+:count_cycles)
	for(int i = 0; i < reduced_graph->Nodes; i++)
	{
		int threadId = omp_get_thread_num();
		count_cycles += multi_work[threadId]->produce_sp_tree_and_cycles(i,reduced_graph);
	}

	std::vector<cycle*> list_cycle_vec;
	std::list<cycle*> list_cycle;

	for(int i=0;i<num_threads;i++)
	{
		for(int j=0;j<multi_work[i]->list_cycles.size();j++)
			list_cycle_vec.push_back(multi_work[i]->list_cycles[j]);
	}

	sort(list_cycle_vec.begin(),list_cycle_vec.end(),cycle::compare());

	for(int i=0; i<list_cycle_vec.size(); i++)
	{
		list_cycle.push_back(list_cycle_vec[i]);
	}

	list_cycle_vec.clear();

	assert(list_cycle.size() == count_cycles);

	printf("List Cycles\n");
	for(std::list<cycle*>::iterator cycle = list_cycle.begin();
			cycle != list_cycle.end(); cycle++)
	{
		printf("%u-(%u - %u) : %d\n",((*cycle))->get_root() + 1,
					reduced_graph->rows->at((*cycle)->non_tree_edge_index) + 1,
					reduced_graph->columns->at((*cycle)->non_tree_edge_index) + 1,
					(*cycle)->total_length);
	}

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
			multi_work[i]->precompute_supportVec(*non_tree_edges_map,*support_vectors[e]);
		}

		for(std::list<cycle*>::iterator cycle = list_cycle.begin();
			cycle != list_cycle.end(); cycle++)
		{
			
			unsigned normal_edge = (*cycle)->non_tree_edge_index;
			unsigned reverse_edge = reduced_graph->reverse_edge->at(normal_edge);
			unsigned bit_val = 0;

			unsigned row,col;
			row = reduced_graph->rows->at(normal_edge);
			col = reduced_graph->columns->at(normal_edge);

			if(non_tree_edges_map->find(reverse_edge) != non_tree_edges_map->end())
			{
				bit_val = support_vectors[e]->get_bit(non_tree_edges_map->at(reverse_edge));
			}
			else if(non_tree_edges_map->find(normal_edge) != non_tree_edges_map->end())
			{
				bit_val = support_vectors[e]->get_bit(non_tree_edges_map->at(normal_edge));
			}

			bit_val = (bit_val + (*cycle)->tree->node_pre_compute->at(row))%2;
			bit_val = (bit_val + (*cycle)->tree->node_pre_compute->at(col))%2;

			if(bit_val == 1)
			{

				final_mcb.push_back(*cycle);
				list_cycle.erase(cycle);
				break;
			}
		}

		bit_vector *cycle_vector = final_mcb.back()->get_cycle_vector(*non_tree_edges_map);
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

	for(int i=0;i<final_mcb.size();i++)
	{
		final_mcb[i]->print();
	}



	return 0;
}