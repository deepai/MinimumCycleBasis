/*
 * bcc_seq.cpp
 *
 *  Created on: 10-Jul-2015
 *      Author: debarshi
 */
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


#include "bicc.h"
#include "dfs.h"
#include "connected_component.h"
#include "dfs_helper.h"
#include "FileReader.h"
#include "Files.h"
#include "utils.h"
#include "Host_Timer.h"
#include "CsrGraph.h"

using namespace std;

debugger dbg;
HostTimer globalTimer;

std::string InputDirectoryName;
std::string OutputFileDirectory;

int global_output_file_count = 0;
int global_edges_removed = 0;
int global_num_bridges = 0;
double totalTime = 0;

int main(int argc,char* argv[])
{
		if(argc < 5)
		{
			printf("Ist Argument should indicate the Inputdirectory\n");
			printf("2nd Argument should indicate the outputdirectory\n");
			printf("3rd Argument should indicate the degree of pruning into bccs.\n");
			printf("4th Argument should indicate the number of nodes.\n");
			printf("5th argument should indicate the number of threads.(Optional) \n");
			exit(1);
		}

		int num_threads = 1;

		if(argc == 6)
			num_threads = atoi(argv[5]);

	    	omp_set_num_threads(num_threads);

	        InputDirectoryName = argv[1];
	        OutputFileDirectory = argv[2];

	  //Obtain the list of the files in the directory.
		std::vector<std::string> fileList = openDirectory(InputDirectoryName);

		int degree_pruning = atoi(argv[3]);
		int global_nodes_count = atoi(argv[4]);

		for(int i=0;i<fileList.size();i++)
		{
			//Open the FileReader class
			string InputFilePath = InputDirectoryName + "/" + fileList[i];

			//Read the Inputfile.
			FileReader Reader(InputFilePath.c_str());

			int v1,v2,Initial_Vertices;;

			int nodes,edges;

			//firt line of the input file contains the number of nodes and edges
			Reader.get_nodes_edges(nodes,edges); 

			bicc_graph *graph=new bicc_graph(nodes);

			/*
			 * ====================================================================================
			 * Fill Edges.
			 * ====================================================================================
			 */
			for(int i=0;i<edges;i++)
			{
				Reader.read_edge(v1,v2);
				graph->insert_edge(v1,v2,false);
			}

			graph->calculate_nodes_edges();
			graph->initialize_bicc_numbers();

			Reader.fileClose();


			debug("Input File Reading Complete...\n");

			/*
			 * ====================================================================================
			 * Each File specific Initialization.
			 * ====================================================================================
			 */
			int component_number = 1;

			int new_component_number = 1;

			//This datastructure is used to hold edge_lists corresponding to each component number
			std::unordered_map<int,std::list<int>* > edge_list_component;

			//This datastructure is used to hold the vertex lists corresponding to each component number.
			std::unordered_map<int,int> src_vtx_component;

			//Initialize the starting source vertex for component 1.
			src_vtx_component[1]=0;

			/*
			 * ====================================================================================
			 * Edge_Map stores the mapping from <src,dest> => edge_index for the initial graph.
			 * ====================================================================================
			 */
			std::unordered_map<unsigned long long,int> *edge_map = create_map(graph->c_graph);

			/*
			 * ====================================================================================
			 * Vector for dfs_helper. Number of elements in dfs_helper = min(number of components,4).
			 * Initially, only one dfs_helper is required.
			 * ====================================================================================
			 */
			std::vector<dfs_helper*> vec_dfs_helper;

			for(int i=0; i<num_threads; i++)
				vec_dfs_helper.push_back(new dfs_helper(global_nodes_count));

			debug("Initialization of the graph completed.\n");

			/*
			 * =========================================================================================
			 * Invoke connected_component for the first run on the Input Graph.
			 * ==========================================================================================
			 */
			int num_components = obtain_connected_components(component_number,new_component_number,graph,
				vec_dfs_helper[0],edge_map);

			debug("Obtained the Initial Connected Components :",num_components);

			edge_list_component.clear();
			src_vtx_component.clear();

			/*
			 * =========================================================================================
			 * For every component within the range [start,end], collect the edge_lists and collect one 
			 * source vertex.
			 * ==========================================================================================
			 */
			graph->collect_edges_component(component_number + 1,new_component_number,edge_list_component,
				src_vtx_component);

			double _counter_init = globalTimer.start_timer();


			assert( !src_vtx_component.empty() );

			bool flag = true;
			/*
			 * =========================================================================================
			 * Follow the above steps in a loop, until we cannot remove any edge from any component.
			 * The finished components are the components whose nodes have degree higher than
			 * the filter threshold.
			 * ==========================================================================================
			 */
			std::unordered_set<int> finished_components;

			std::vector<int> component_list;

			for(std::unordered_map<int,std::list<int>* >::iterator it=edge_list_component.begin();
				it!=edge_list_component.end();it++)
			{
				component_list.push_back(it->first);
			}

			int num_iterations = 0;

			double time_dfs = 0;
			double time_pruning = 0;

			while( flag )
			{
				num_iterations++;

				flag = false;

				component_number = new_component_number;

				edge_list_component.clear();

				double _local_time_dfs = globalTimer.start_timer();

				#pragma omp parallel for
				for(int i=0; i<component_list.size(); i++)
				{
					int thread_id = omp_get_thread_num();

					if(finished_components.find(component_list[i]) != finished_components.end())
						continue;

					//debug("Active component DFS:",component_list[i],src_vtx_component[component_list[i]] + 1);

					int num_bridges = dfs_bicc_initializer(src_vtx_component[component_list[i]],
						component_list[i],new_component_number,graph,vec_dfs_helper[thread_id],
						edge_map,edge_list_component);


					global_num_bridges += num_bridges;
				}

				time_dfs += (globalTimer.stop_timer() - _local_time_dfs);

				src_vtx_component.clear();


				/*
				 * ====================================================================
				 * Structures required for parallel computation. 
				 * Each omp thread works on an individual component for the pruning part.
				 * For each thread. we add the component numbers which cannot be further pruned to the 
				 * list_finished_components.
				 * 
				 * component_list just contains the list of component numbers collected from the 
				 * above run.
				 * ====================================================================
				 */
				std::list<int> list_finished_components[num_threads]; //This list is used to hold the finished component numbers for num_threads

				component_list.clear();

				for(std::unordered_map<int,std::list<int>* >::iterator it=edge_list_component.begin();					
					it!=edge_list_component.end();it++)
				{
					int edge_end_point = graph->c_graph->rows->at(it->second->front());
					src_vtx_component[it->first] = edge_end_point;
					
					component_list.push_back(it->first);
				}

				//debug("Size of Component_list:",component_list.size());

				/*
				 * ====================================================================
				 * Parallely prune the edge lists and update the finished component 
				 * ====================================================================
				 */
				 double _local_time_pruning = globalTimer.start_timer();

				#pragma omp parallel for
				for(int i=0; i<component_list.size(); i++)
				{
					int thread_id = omp_get_thread_num();

					//debug("Active component Prune:",component_list[i],src_vtx_component[component_list[i]] + 1);

					if( finished_components.find(component_list[i]) != finished_components.end() )
						continue;

					int num_edges = 0;
					num_edges += graph->prune_edges(degree_pruning,component_list[i],
						edge_list_component[component_list[i]],edge_map,src_vtx_component);

					if(num_edges == 0)
						list_finished_components[thread_id].push_back(component_list[i]);

					global_edges_removed += num_edges;

					//debug("number of edges removed:",thread_id,num_edges);

					if(num_edges != 0)
					{
					#pragma omp critical
						flag = 1;
					}
				}

				time_pruning += (globalTimer.stop_timer() - _local_time_pruning);

				/*
				 * ====================================================================
				 * Insert the finished component Ids into the finished components list.
				 * ====================================================================
				 */
				for(int i=0; i<num_threads ;i++)
				{
					for(std::list<int>::iterator it=list_finished_components[i].begin();
						it!=list_finished_components[i].end();it++)
					{
						finished_components.insert(*it);
					}
					list_finished_components[i].clear();
				}


			}

			debug("Num Iterations:",num_iterations);

			double _counter_exit = globalTimer.stop_timer();

			totalTime += (_counter_exit - _counter_init);

			debug("Total Number of Components in the current file =",finished_components.size());

			graph->print_to_a_file(global_output_file_count,OutputFileDirectory,global_nodes_count,finished_components);

			delete graph;
			edge_list_component.clear();
			src_vtx_component.clear();
			vec_dfs_helper.clear();

			debug("Total dfs time:",time_dfs);
			debug("Total pruning time:",time_pruning);

		}
		
		debug("Total Edges Removed:",global_edges_removed);
		debug("Total Number of Bridges",global_num_bridges);
		debug("Total Number of components",global_output_file_count);

		printf("%d\n",global_edges_removed);
		printf("%lf\n",totalTime);

		return 0;
}




