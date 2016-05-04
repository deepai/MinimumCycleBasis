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
#include <cstring>
#include <unordered_set>
#include <unordered_map>
#include <utility>

#include "FileReader.h"
#include "Files.h"
#include "utils.h"
#include "Host_Timer.h"
#include "CsrGraph.h"
#include "CsrTree.h"
#include "CsrGraphMulti.h"
#include "bit_vector.h"
#include "work_per_thread.h"
#include "cycle_searcher.h"
#include "isometric_cycle.h"
#include "stats.h"

debugger dbg;
HostTimer globalTimer;

std::string InputFileName;
std::string OutputFileDirectory;

double totalTime = 0;
double localTime = 0;

stats info;

int main(int argc,char* argv[])
{
	if(argc < 4)
	{
		printf("Ist Argument should indicate the InputFile\n");
		printf("2nd Argument should indicate the OutputFile\n");
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
	graph->initial_edge_count = edges;
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
		info.setNumFinalCycles(1);
		info.setNumInitialCycles(1);
		info.setTotalWeight(graph->get_total_weight());
		info.print_stats(argv[2]);
		
		return 0;
	}

	int source_vertex = 0;

	csr_multi_graph *reduced_graph = csr_multi_graph::get_modified_graph(graph,
									     NULL,
									     NULL,
									     0);

	csr_tree *initial_spanning_tree = new csr_tree(reduced_graph);
	initial_spanning_tree->populate_tree_edges(true,source_vertex);

	int num_non_tree_edges = initial_spanning_tree->non_tree_edges->size();

	assert(num_non_tree_edges == edges - nodes + 1);
	assert(graph->get_total_weight() == reduced_graph->get_total_weight());

	std::vector<std::pair<bool,int>> non_tree_edges_map(reduced_graph->rows->size());
	
	for(int i=0;i<initial_spanning_tree->non_tree_edges->size();i++)
	{
		non_tree_edges_map[initial_spanning_tree->non_tree_edges->at(i)].first = true;
		non_tree_edges_map[initial_spanning_tree->non_tree_edges->at(i)].second = i;
	}

	for(int i=0;i<reduced_graph->rows->size();i++)
	{
		//copy the edges into the reverse edges as well.
		if(!non_tree_edges_map[i].first)
		{
			if(non_tree_edges_map[reduced_graph->reverse_edge->at(i)].first)
			{
				non_tree_edges_map[i].first = true;
				non_tree_edges_map[i].second = non_tree_edges_map[reduced_graph->reverse_edge->at(i)].second;
			}
		}
	}

	cycle_storage *storage = new cycle_storage(reduced_graph->Nodes);

	worker_thread **multi_work = new worker_thread*[num_threads];

	for(int i=0;i<num_threads;i++)
		multi_work[i] = new worker_thread(reduced_graph,storage);

	globalTimer.start_timer();

	//produce shortest path trees across all the nodes.

	int count_cycles = 0;

	#pragma omp parallel for reduction(+:count_cycles)
	for(int i = 0; i < reduced_graph->Nodes; ++i)
	{
		int threadId = omp_get_thread_num();
		count_cycles += multi_work[threadId]->produce_sp_tree_and_cycles(i,reduced_graph);
	}

	for(int i=0;i<num_threads;i++)
		storage->add_trees(multi_work[i]->shortest_path_trees);

	localTime = globalTimer.get_event_time();
	totalTime += localTime;

	info.setTimeConstructionTrees(localTime);

	globalTimer.start_timer();

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

	info.setNumInitialCycles(list_cycle_vec.size());

	isometric_cycle *isometric_cycle_helper = new isometric_cycle(list_cycle_vec.size(),storage,&list_cycle_vec);

	isometric_cycle_helper->obtain_isometric_cycles();

	delete isometric_cycle_helper;

	for(int i=0; i<list_cycle_vec.size(); i++)
	{
		if(list_cycle_vec[i] != NULL)
			list_cycle.push_back(list_cycle_vec[i]);
	}

	info.setNumIsometricCycles(list_cycle.size());

	list_cycle_vec.clear();


	//assert(list_cycle.size() == count_cycles);

	localTime = globalTimer.get_event_time();
	totalTime += localTime;

	info.setTimeCollectCycles(localTime);

	//At this stage we have the shortest path trees and the cycles sorted in increasing order of length.

	//generate the bit vectors
	bit_vector **support_vectors = new bit_vector*[num_non_tree_edges];
	for(int i=0;i<num_non_tree_edges;i++)
	{
		support_vectors[i] = new bit_vector(num_non_tree_edges);
		support_vectors[i]->set_bit(i,true);
	}

	std::vector<cycle*> final_mcb;

	double precompute_time = 0;
	double cycle_inspection_time = 0;
	double independence_test_time = 0;

	//Main Outer Loop of the Algorithm.
	for(int e=0;e<num_non_tree_edges;e++)
	{
		globalTimer.start_timer();

		#pragma omp parallel for
		for(int i=0;i<num_threads;i++)
		{
			multi_work[i]->precompute_supportVec(non_tree_edges_map,*support_vectors[e]);
		}

		precompute_time += globalTimer.get_event_time();
		globalTimer.start_timer();


		for(std::list<cycle*>::iterator cycle = list_cycle.begin();
			cycle != list_cycle.end(); cycle++)
		{
			
			unsigned normal_edge = (*cycle)->non_tree_edge_index;
			unsigned bit_val = 0;

			unsigned row,col;
			row = reduced_graph->rows->at(normal_edge);
			col = reduced_graph->columns->at(normal_edge);

			std::pair<bool,int> &edge = non_tree_edges_map[normal_edge];

			if(edge.first)
			{
				bit_val = support_vectors[e]->get_bit(edge.second);
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

		cycle_inspection_time += globalTimer.get_event_time();
		globalTimer.start_timer();

		bit_vector *cycle_vector = final_mcb.back()->get_cycle_vector(non_tree_edges_map,
																	  initial_spanning_tree->non_tree_edges->size());

		for(int j=e+1;j<num_non_tree_edges;j++)
		{
			unsigned product = cycle_vector->dot_product(support_vectors[j]);
			if(product == 1)
				support_vectors[j]->do_xor(support_vectors[e]);
		}

		independence_test_time += globalTimer.get_event_time();

	}

	list_cycle.clear();

	info.setPrecomputeShortestPathTime(precompute_time);
	info.setCycleInspectionTime(cycle_inspection_time);
	info.setIndependenceTestTime(independence_test_time);
	info.setTotalTime();

	int total_weight = 0;

	for(int i=0;i<final_mcb.size();i++)
	{
		total_weight +=  final_mcb[i]->total_length;
	}

	info.setNumFinalCycles(final_mcb.size());
	info.setTotalWeight(total_weight);

	info.print_stats(argv[2]);

	return 0;
}