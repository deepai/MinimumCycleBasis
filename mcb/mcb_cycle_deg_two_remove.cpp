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
#include "stats.h"
#include "FVS.h"
#include "compressed_trees.h"
#include "gpu_task.h"

#include <gpu/common.cuh>

debugger dbg;
HostTimer globalTimer;

std::string InputFileName;
std::string OutputFileDirectory;

double totalTime = 0;
double localTime = 0;

stats info;

int num_threads;

int main(int argc, char* argv[]) {
	if (argc < 4) {
		printf("Ist Argument should indicate the InputFile\n");
		printf("2nd Argument should indicate the OutputFile\n");
		printf(
				"3th argument should indicate the number of threads.(Optional) (1 default)\n");
		exit(1);
	}

	num_threads = 1;

	if (argc == 4)
		num_threads = atoi(argv[3]);

	InputFileName = argv[1];

	omp_set_num_threads(num_threads);

	//Open the FileReader class
	std::string InputFilePath = InputFileName;

	//Read the Inputfile.
	FileReader Reader(InputFilePath.c_str());

	int v1, v2, Initial_Vertices, weight;
	;

	int nodes, edges, chunk_size = 1, nstreams = 1;

	//firt line of the input file contains the number of nodes and edges
	Reader.get_nodes_edges(nodes, edges);

	csr_graph *graph = new csr_graph();

	graph->Nodes = nodes;
	graph->initial_edge_count = edges;
	/*
	 * ====================================================================================
	 * Fill Edges.
	 * ====================================================================================
	 */
	for (int i = 0; i < edges; i++) {
		Reader.read_edge(v1, v2, weight);
		graph->insert(v1, v2, weight, false);
	}

	graph->calculateDegreeandRowOffset();

	info.setNumNodesTotal(graph->Nodes);

	Reader.fileClose();

	if (graph->get_num_degree_two_vertices() == graph->Nodes) {
		info.setCycleNumFVS(1);
		info.setNumNodesRemoved(graph->Nodes - 1);
		info.setNumFinalCycles(1);
		info.setNumInitialCycles(1);
		info.setTotalWeight(graph->get_total_weight());
		info.print_stats(argv[2]);

		return 0;
	}

	init_cuda();

	std::vector<std::vector<unsigned> > *chains = new std::vector<
			std::vector<unsigned> >();

	int source_vertex;

	std::vector<unsigned> *remove_edge_list = graph->mark_degree_two_chains(
			&chains, source_vertex);
	//initial_spanning_tree.populate_tree_edges(true,NULL,source_vertex);

	std::vector<std::vector<unsigned> > *edges_new_list = new std::vector<
			std::vector<unsigned> >();

	int nodes_removed = 0;

	for (int i = 0; i < chains->size(); i++) {
		unsigned row, col;
		unsigned total_weight = graph->sum_edge_weights(chains->at(i), row,
				col);

		nodes_removed += chains->at(i).size() - 1;

		std::vector<unsigned> new_edge = std::vector<unsigned>();
		new_edge.push_back(row);
		new_edge.push_back(col);
		new_edge.push_back(total_weight);

		edges_new_list->push_back(new_edge);
		//debug(row+1,col+1,total_weight);
	}

	assert(nodes_removed == graph->get_num_degree_two_vertices());

	info.setNumNodesRemoved(nodes_removed);

	csr_multi_graph *reduced_graph = csr_multi_graph::get_modified_graph(graph,
			remove_edge_list, edges_new_list, nodes_removed);

	FVS fvs_helper(reduced_graph);
	fvs_helper.MGA();
	fvs_helper.print_fvs();

	info.setCycleNumFVS(fvs_helper.get_num_elements());

	int *fvs_array = fvs_helper.get_copy_fvs_array();

	csr_tree *initial_spanning_tree = new csr_tree(reduced_graph);
	initial_spanning_tree->populate_tree_edges(true, source_vertex);

	int num_non_tree_edges = initial_spanning_tree->non_tree_edges->size();

	assert(num_non_tree_edges == edges - nodes + 1);
	assert(graph->get_total_weight() == reduced_graph->get_total_weight());

	std::vector<int> non_tree_edges_map(reduced_graph->rows->size());
	std::fill(non_tree_edges_map.begin(), non_tree_edges_map.end(), -1);

	for (int i = 0; i < initial_spanning_tree->non_tree_edges->size(); i++)
		non_tree_edges_map[initial_spanning_tree->non_tree_edges->at(i)] = i;

	for (int i = 0; i < reduced_graph->rows->size(); i++) {
		//copy the edges into the reverse edges as well.
		if (non_tree_edges_map[i] < 0)
			if (non_tree_edges_map[reduced_graph->reverse_edge->at(i)] >= 0)
				non_tree_edges_map[i] =
						non_tree_edges_map[reduced_graph->reverse_edge->at(i)];
	}

	calculate_chunk_size(reduced_graph->Nodes, non_tree_edges_map.size(),
			(int) (ceil((double) num_non_tree_edges / 64)), 32);

	chunk_size = (int)(ceil((double)fvs_helper.get_num_elements()/nstreams));

	//construct the initial
	//compressed_trees trees(chunk_size,fvs_helper.get_num_elements(),fvs_array,reduced_graph);
	compressed_trees trees(chunk_size, fvs_helper.get_num_elements(), fvs_array,
			reduced_graph, allocate_pinned_memory, free_pinned_memory);

	cycle_storage *storage = new cycle_storage(reduced_graph->Nodes);

	worker_thread **multi_work = new worker_thread*[num_threads];

	for (int i = 0; i < num_threads; i++)
		multi_work[i] = new worker_thread(reduced_graph, storage, fvs_array,
				&trees);

	globalTimer.start_timer();

	//produce shortest path trees across all the nodes.

	int count_cycles = 0;

#pragma omp parallel for reduction(+:count_cycles)
	for (int i = 0; i < trees.fvs_size; ++i) {
		int threadId = omp_get_thread_num();
		count_cycles += multi_work[threadId]->produce_sp_tree_and_cycles_warp(i,
				reduced_graph);
	}

	localTime = globalTimer.get_event_time();
	totalTime += localTime;

	info.setTimeConstructionTrees(localTime);

	globalTimer.start_timer();

	std::vector<cycle*> list_cycle_vec;
	std::list<cycle*> list_cycle;

	for (int j = 0; j < storage->list_cycles.size(); j++) {
		for (std::unordered_map<unsigned long long, list_common_cycles*>::iterator it =
				storage->list_cycles[j].begin();
				it != storage->list_cycles[j].end(); it++) {
			for (int k = 0; k < it->second->listed_cycles.size(); k++) {
				list_cycle_vec.push_back(it->second->listed_cycles[k]);
				list_cycle_vec.back()->ID = list_cycle_vec.size() - 1;
			}
		}
	}

	sort(list_cycle_vec.begin(), list_cycle_vec.end(), cycle::compare());

	info.setNumInitialCycles(list_cycle_vec.size());

	for (int i = 0; i < list_cycle_vec.size(); i++) {
		if (list_cycle_vec[i] != NULL)
			list_cycle.push_back(list_cycle_vec[i]);
	}

	info.setNumIsometricCycles(list_cycle.size());

	list_cycle_vec.clear();

	localTime = globalTimer.get_event_time();
	totalTime += localTime;

	info.setTimeCollectCycles(localTime);

	//At this stage we have the shortest path trees and the cycles sorted in increasing order of length.

	//generate the bit vectors
	bit_vector **support_vectors = new bit_vector*[num_non_tree_edges];
	for (int i = 0; i < num_non_tree_edges; i++) {
		support_vectors[i] = new bit_vector(num_non_tree_edges);
		support_vectors[i]->set_bit(i, true);
	}

	gpu_task gpu_compute(&trees, (int*) trees.final_vertices,
			non_tree_edges_map, support_vectors, num_non_tree_edges);
	gpu_struct device_struct(non_tree_edges_map.size(), num_non_tree_edges,
			support_vectors[0]->size, gpu_compute.original_nodes,
			gpu_compute.fvs_size, chunk_size, nstreams);

	configure_grid(0, gpu_compute.fvs_size);

	device_struct.initialize_memory(&gpu_compute);
	device_struct.calculate_memory();

	std::vector<cycle*> final_mcb;

	double precompute_time = 0;
	double cycle_inspection_time = 0;
	double independence_test_time = 0;
	double gpu_precompute_time = 0;

	double kernel_init_time = 0;
	double kernel_multi_search_time = 0;
	double transfer_time = 0;

	bit_vector *cycle_vector = new bit_vector(num_non_tree_edges,
			allocate_pinned_memory, free_pinned_memory);
	bit_vector *current_vector = new bit_vector(num_non_tree_edges,
			allocate_pinned_memory, free_pinned_memory);

	current_vector->init_zero();
	current_vector->set_bit(0, true);

	//Main Outer Loop of the Algorithm.
	for (int e = 0; e < num_non_tree_edges; e++) {
		//globalTimer.start_timer();

		transfer_time += device_struct.copy_support_vector(current_vector);

		kernel_init_time += device_struct.Kernel_init_edges_helper(0,
				gpu_compute.fvs_size, 0);
		kernel_multi_search_time += device_struct.Kernel_multi_search_helper(0,
				gpu_compute.fvs_size, 0);

		transfer_time += device_struct.fetch(&gpu_compute);

		globalTimer.start_timer();

		unsigned *node_rowoffsets, *node_columns, *precompute_nodes,
				*nodes_index;
		int *node_edgeoffsets, *node_parents, *node_distance;
		unsigned src, edge_offset, reverse_edge, row, col, position, bit;

		int src_index;

		for (std::list<cycle*>::iterator cycle = list_cycle.begin();
				cycle != list_cycle.end(); cycle++) {
			src = (*cycle)->get_root();
			src_index = trees.vertices_map[src];

			trees.get_node_arrays_warp(&node_rowoffsets, &node_columns,
					&node_edgeoffsets, &node_parents, &node_distance,
					&nodes_index, src_index);
			trees.get_precompute_array(&precompute_nodes, src_index);

			edge_offset = (*cycle)->non_tree_edge_index;
			bit = 0;

			unsigned row, col;
			row = reduced_graph->rows->at(edge_offset);
			col = reduced_graph->columns->at(edge_offset);

			if (non_tree_edges_map[edge_offset] >= 0) {
				bit = current_vector->get_bit(non_tree_edges_map[edge_offset]);
			}

			bit = (bit ^ precompute_nodes[nodes_index[row]]);
			bit = (bit ^ precompute_nodes[nodes_index[col]]);

			if (bit == 1) {
				final_mcb.push_back(*cycle);
				list_cycle.erase(cycle);
				break;
			}
		}

		cycle_inspection_time += globalTimer.get_event_time();
		globalTimer.start_timer();

		final_mcb.back()->get_cycle_vector(non_tree_edges_map,
				initial_spanning_tree->non_tree_edges->size(), cycle_vector);

#pragma omp parallel for
		for (int j = e + 1; j < num_non_tree_edges; j++) {
			unsigned product = cycle_vector->dot_product(support_vectors[j]);
			if (product == 1)
				support_vectors[j]->do_xor(current_vector);
		}

		if (e < num_non_tree_edges - 1)
			current_vector->copy_vector(support_vectors[e + 1]);

		independence_test_time += globalTimer.get_event_time();

	}

	cycle_vector->clear_memory();
	current_vector->clear_memory();

	list_cycle.clear();

	gpu_precompute_time = kernel_init_time + kernel_multi_search_time
			+ transfer_time;

	info.setPrecomputeShortestPathTime(gpu_precompute_time / 1000);
	info.setCycleInspectionTime(cycle_inspection_time);
	info.setIndependenceTestTime(independence_test_time);
	info.setTotalTime();

	int total_weight = 0;

	for (int i = 0; i < final_mcb.size(); i++) {
		total_weight += final_mcb[i]->total_length;
	}

	info.setNumFinalCycles(final_mcb.size());
	info.setTotalWeight(total_weight);

	info.print_stats(argv[2]);

	delete[] fvs_array;

	debug(kernel_init_time/1000);
	debug(kernel_multi_search_time/1000);
	debug(transfer_time/1000);
	debug(gpu_precompute_time/1000);

	device_struct.clear_memory();
	trees.clear_memory();

	for (int i = 0; i < num_non_tree_edges; i++)
		support_vectors[i]->clear_memory();

	delete[] support_vectors;

	return 0;
}
