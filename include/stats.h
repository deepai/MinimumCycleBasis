#ifndef _STATS_H
#define _STATS_H

#include <iostream>
#include <cstdio>
#include <unistd.h>

struct stats
{
	int num_nodes_removed;
	int num_initial_cycles;
	int num_isometric_cycles;
	int num_nodes;

	int num_fvs = 0;

	int num_final_cycles;
	int total_weight;

	double time_construction_trees;
	double time_collect_cycles;
	double precompute_shortest_path_time;
	double cycle_inspection_time;
	double independence_test_time;

	double total_time = 0;

	stats()
	{
		num_nodes_removed = 0;
		num_initial_cycles = 0;
		num_isometric_cycles = 0;
		num_final_cycles = 0;
		total_weight = 0;

		num_fvs = 0;
		num_nodes = 0;

		time_construction_trees = 0;
		time_collect_cycles = 0;
		precompute_shortest_path_time = 0;
		cycle_inspection_time = 0;
		independence_test_time = 0;

		total_time = 0;
	}

	void stats::setNumNodesTotal(int num_nodes_total)
	{
		num_nodes = num_nodes_total;
	}

	void stats::setCycleNumFVS(int numfvs)
	{
		num_fvs = numfvs;
	}

	void stats::setCycleInspectionTime(double cycleInspectionTime) {
		cycle_inspection_time = cycleInspectionTime;
	}

	void stats::setIndependenceTestTime(double independenceTestTime) {
		independence_test_time = independenceTestTime;
	}

	void stats::setNumFinalCycles(int numFinalCycles) {
		num_final_cycles = numFinalCycles;
	}

	void stats::setNumInitialCycles(int numInitialCycles) {
		num_initial_cycles = numInitialCycles;
	}

	void stats::setNumIsometricCycles(int numIsometricCycles) {
		num_isometric_cycles = numIsometricCycles;
	}

	void stats::setNumNodesRemoved(int numNodesRemoved) {
		num_nodes_removed = numNodesRemoved;
	}

	void stats::setPrecomputeShortestPathTime(double precomputeShortestPathTime) {
		precompute_shortest_path_time = precomputeShortestPathTime;
	}

	void stats::setTimeCollectCycles(double timeCollectCycles) {
		time_collect_cycles = timeCollectCycles;
	}

	void stats::setTimeConstructionTrees(double timeConstructionTrees) {
		time_construction_trees = timeConstructionTrees;
	}

	void stats::setTotalTime(double totalTime = 0) {
		total_time = precompute_shortest_path_time + independence_test_time + cycle_inspection_time;
	}

	void stats::setTotalWeight(int totalWeight) {
		total_weight = totalWeight;
	}


	void print_stats(char *output_file)
	{
		bool file_exist = false;

		if( access( output_file, F_OK ) != -1 ) {
    		file_exist = true;
		} else {
    		file_exist = false;
		}

		FILE *fout = fopen(output_file,"a");

		if(!file_exist)
		{
			fprintf(fout,"Nodes Removed,FVS size,Total_Vertices_Removed,Initial Cycles,Isometric_cycles,final_cycles,Total_Weight,construction_trees(s),collect_cycles(s),inspection_time(s),precompute_shortest_path(s),independence_test(s),total_time(s)\n");
		}

		fprintf(fout,"%5d,%5d,%5d,%5d,%5d,%5d,%5d,%15lf,%15lf,%15lf,%15lf,%15lf,%15lf\n",num_nodes_removed,
																								num_fvs,
																								num_nodes + num_nodes_removed - num_fvs,
																								num_initial_cycles,
																								num_isometric_cycles,
																								num_final_cycles,
																								total_weight,
																								time_construction_trees,
																								time_collect_cycles,
																								cycle_inspection_time,
																								precompute_shortest_path_time,
																								independence_test_time,
																								total_time);


		fclose(fout);
	}

};


#endif