#ifndef _STATS_H
#define _STATS_H

#include <iostream>
#include <cstdio>
#include <unistd.h>

struct stats {
	int num_nodes_removed;
	int num_initial_cycles;
	int num_nodes;

	int num_fvs = 0;
	int edges = 0;

	int new_edges = 0;

	int num_final_cycles;
	int total_weight;

	double time_construction_trees;
	double time_collect_cycles;
	double precompute_shortest_path_time;
	double cycle_inspection_time;
	double independence_test_time;

	double total_time = 0;

	//GPU STATS
	int nchunks;
	int nstreams;
	double total_memory_usage;
	double static_memory_usage;
	double variable_memory_usage;

	bool is_gpu_timings;

	double gpu_timings;
	bool load_entire_memory;

	stats(bool is_gpu) {

		is_gpu_timings = is_gpu;
		num_nodes_removed = 0;
		num_initial_cycles = 0;
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

		nchunks = 0;
		nstreams = 0;
		total_memory_usage = 0;
		static_memory_usage = 0;
		variable_memory_usage = 0;

		gpu_timings = 0;

		load_entire_memory = true;
	}

	void setNumNodesTotal(int num_nodes_total) {
		num_nodes = num_nodes_total;
	}

	void setCycleNumFVS(int numfvs) {
		num_fvs = numfvs;
	}

	void setCycleInspectionTime(double cycleInspectionTime) {
		cycle_inspection_time = cycleInspectionTime;
	}

	void setIndependenceTestTime(double independenceTestTime) {
		independence_test_time = independenceTestTime;
	}

	void setNumFinalCycles(int numFinalCycles) {
		num_final_cycles = numFinalCycles;
	}

	void setNumInitialCycles(int numInitialCycles) {
		num_initial_cycles = numInitialCycles;
	}

	void setNumNodesRemoved(int numNodesRemoved) {
		num_nodes_removed = numNodesRemoved;
	}

	void setPrecomputeShortestPathTime(double precomputeShortestPathTime) {
		precompute_shortest_path_time = precomputeShortestPathTime;
	}

	void setTimeCollectCycles(double timeCollectCycles) {
		time_collect_cycles = timeCollectCycles;
	}

	void setTimeConstructionTrees(double timeConstructionTrees) {
		time_construction_trees = timeConstructionTrees;
	}

	void setTotalTime(double totalTime = 0) {

		if (!is_gpu_timings)
			total_time = precompute_shortest_path_time + independence_test_time
					+ cycle_inspection_time;
		else
			total_time = independence_test_time + cycle_inspection_time;
	}

	void setTotalWeight(int totalWeight) {
		total_weight = totalWeight;
	}

	double getCycleInspectionTime() const {
		return cycle_inspection_time;
	}

	int getEdges() const {
		return edges;
	}

	void setEdges(int edges = 0) {
		this->edges = edges;
	}

	void setNewEdges(int new_edges = 0) {
		this->new_edges = new_edges;
	}

	double getGpuTimings() const {
		return gpu_timings;
	}

	void setGpuTimings(double gpuTimings) {
		gpu_timings = gpuTimings;
	}

	double getIndependenceTestTime() const {
		return independence_test_time;
	}

	bool isIsGpuTimings() const {
		return is_gpu_timings;
	}

	void setIsGpuTimings(bool isGpuTimings) {
		is_gpu_timings = isGpuTimings;
	}

	bool isLoadEntireMemory() const {
		return load_entire_memory;
	}

	void setLoadEntireMemory(bool loadEntireMemory) {
		load_entire_memory = loadEntireMemory;
	}

	int getNchunks() const {
		return nchunks;
	}

	void setNchunks(int nchunks) {
		this->nchunks = nchunks;
	}

	int getNstreams() const {
		return nstreams;
	}

	void setNstreams(int nstreams) {
		this->nstreams = nstreams;
	}

	int getNumFinalCycles() const {
		return num_final_cycles;
	}

	int getNumFvs() const {
		return num_fvs;
	}

	void setNumFvs(int numFvs = 0) {
		num_fvs = numFvs;
	}

	int getNumInitialCycles() const {
		return num_initial_cycles;
	}

	int getNumNodes() const {
		return num_nodes;
	}

	void setNumNodes(int numNodes) {
		num_nodes = numNodes;
	}

	int getNumNodesRemoved() const {
		return num_nodes_removed;
	}

	double getPrecomputeShortestPathTime() const {
		return precompute_shortest_path_time;
	}

	double getStaticMemoryUsage() const {
		return static_memory_usage;
	}

	void setStaticMemoryUsage(double staticMemoryUsage) {
		static_memory_usage = staticMemoryUsage;
	}

	double getTimeCollectCycles() const {
		return time_collect_cycles;
	}

	double getTimeConstructionTrees() const {
		return time_construction_trees;
	}

	double getTotalMemoryUsage() const {
		return total_memory_usage;
	}

	void setTotalMemoryUsage(double totalMemoryUsage) {
		total_memory_usage = totalMemoryUsage;
	}

	double getTotalTime() const {
		return total_time;
	}

	int getTotalWeight() const {
		return total_weight;
	}

	double getVariableMemoryUsage() const {
		return variable_memory_usage;
	}

	void setVariableMemoryUsage(size_t variableMemoryUsage) {
		variable_memory_usage = variableMemoryUsage;
	}

	void print_stats(char *output_file) {
		bool file_exist = false;

		if (access(output_file, F_OK) != -1) {
			file_exist = true;
		} else {
			file_exist = false;
		}

		FILE *fout = fopen(output_file, "a");

		if (!file_exist) {

			if (!is_gpu_timings)
				fprintf(fout,
						"Total_Nodes,\
							  Total_Edges,\
							  New_Edges,\
							  Node_removed,\
							  Fvs_size,\
							  Initial_Cycles,\
							  Final_cycles,\
							  Total_Weight,\
							  Construction_trees(s),\
							  Collect_cycles(s),\
							  Inspection_time(s),\
							  Precompute_SP(s),\
							  Independence_test(s),\
							  Preprocessing Time(s),\
							  Main_loop(s),\
							  Total_time(s)\n");
			else
				fprintf(fout,
						"Total_Nodes,\
							  Total_Edges,\
							  New_Edges,\
							  Nodes_removed,\
							  Fvs_size,\
							  Initial_Cycles,\
							  Final_cycles,\
							  Total_Weight,\
							  nchunks,\
							  nstreams,\
							  total_memory_usage(mb),\
							  static_memory_usage(mb),\
							  variable_memory_usage(mb),\
							  load_entire_memory,\
							  Construction_trees(s),\
							  Collect_cycles(s),\
							  Inspection_time(s),\
							  Gpu_timings,\
							  Hybrid_Timings(s),\
							  Preprocessing Time(s),\
							  Main_loop(s),\
							  Total_time(s)\n");
		}

		if (!is_gpu_timings)
			fprintf(fout,
					"%5d,%5d,%5d,%5d,%5d,%5d,%5d,%5d,%15lf,%15lf,%15lf,%15lf,%15lf,%15lf,%15lf,%15lf\n",
					num_nodes, edges, new_edges, num_nodes_removed, num_fvs,
					num_initial_cycles, num_final_cycles, total_weight,
					time_construction_trees, time_collect_cycles,
					cycle_inspection_time, precompute_shortest_path_time,
					independence_test_time,
					time_construction_trees + time_collect_cycles, total_time,
					total_time + time_construction_trees + time_collect_cycles);

		else
			fprintf(fout,
					"%5d,%5d,%5d,%5d,%5d,%5d,%5d,%5d,%5d,%5d,%15lf,%15lf,%15lf,%s,\
					%15lf,%15lf,%15lf,%15lf,%15lf,%15lf,%15lf,%15lf\n",
					num_nodes, edges, new_edges, num_nodes_removed, num_fvs,
					num_initial_cycles, num_final_cycles, total_weight, nchunks,
					nstreams, total_memory_usage, static_memory_usage,
					variable_memory_usage, load_entire_memory ? "YES" : "NO",
					time_construction_trees, time_collect_cycles,
					cycle_inspection_time, gpu_timings, independence_test_time,
					time_construction_trees + time_collect_cycles, total_time,
					total_time + time_construction_trees + time_collect_cycles);

		fclose(fout);
	}
};

#endif
