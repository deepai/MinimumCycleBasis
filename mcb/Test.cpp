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

#include "FileReader.h"
#include "Files.h"
#include "utils.h"
#include "Host_Timer.h"
#include "CsrGraph.h"
#include "CsrTree.h"

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

	debug("Input File Reading Complete...\n");

	int source_vertex;

	std::vector<std::vector<unsigned> > *chains = new std::vector<std::vector<unsigned> >();
	std::vector<unsigned> *remove_edge_list = graph->mark_degree_two_chains(&chains,source_vertex);

	csr_tree initial_spanning_tree(graph);
	std::vector<unsigned> *ear_decomposition = new std::vector<unsigned>(graph->Nodes + 1);

	//initial_spanning_tree.populate_tree_edges(true,ear_decomposition,source_vertex);

	debug("Generating Initial Spanning Tree and Ear Decomposition");

	debug("Number of Ears = ",ear_decomposition->at(graph->Nodes));

	debug("Tree - Edges");

	unsigned row;
	unsigned col;

	// for(int i=0;i<initial_spanning_tree.tree_edges->size();i++)
	// {
	// 	initial_spanning_tree.get_edge_endpoints(row,col,weight,
	// 						 initial_spanning_tree.tree_edges->at(i));
	// 	debug (row + 1,'-',col + 1,weight);
	// }

	// debug("Non-Tree Edges");

	// for(int i=0;i<initial_spanning_tree.non_tree_edges->size();i++)
	// {
	// 	initial_spanning_tree.get_edge_endpoints(row,col,weight,
	// 						 initial_spanning_tree.non_tree_edges->at(i));
	// 	debug (row + 1,'-',col + 1,weight);
	// }

	// debug("Ear Decomposition");

	// for(int i=0;i<ear_decomposition->size() - 1;i++)
	// 	debug(i+1,ear_decomposition->at(i));

	// debug("Number of degree 2 cycles =",chains->size());

	// for(int i=0;i<chains->size();i++)
	// {
	// 	for(int j=0;j<chains->at(i).size();j++)
	// 	{
	// 		//printf("%u ",chains->at(i)[j]);
	// 		unsigned offset = chains->at(i)[j];
	// 		graph->get_edge_endpoints(row,col,weight,offset);
	// 		printf("%u %u\n",row+1,col+1);
	// 	}
	// 	debug("");
	// }

	// debug ("Removed Edges");

	// for(int i=0;i<remove_edge_list->size();i++)
	// {
	// 	unsigned offset = remove_edge_list->at(i);
	// 	graph->get_edge_endpoints(row,col,weight,offset);

	// 	printf("%u - %u : %u\n",row+1,col+1,weight);
	// }

	 return 0;
}