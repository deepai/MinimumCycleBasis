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

	std::vector<std::vector<unsigned> > *chains = new std::vector<std::vector<unsigned> >();

	debug("Input File Reading Complete...\n");
	debug("Generating Initial Spanning Tree and Ear Decomposition");

	csr_tree initial_spanning_tree(graph);

	int souce_vertex;

	std::vector<unsigned> *remove_edge_list = graph->mark_degree_two_chains(&chains,souce_vertex);
	initial_spanning_tree.populate_tree_edges(true,NULL,souce_vertex);

	return 0;
}