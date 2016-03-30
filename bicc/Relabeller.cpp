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
#include <unordered_map>

#include "FileReader.h"
#include "Files.h"
#include "utils.h"
#include "Host_Timer.h"
#include "CsrGraph.h"
#include "CsrTree.h"

debugger dbg;
HostTimer globalTimer;

std::string InputFileName;
std::string OutputFileName;

double totalTime = 0;

std::unordered_map<unsigned,unsigned> forward_order;

int main(int argc,char* argv[])
{
	if(argc < 4)
	{
		printf("Ist Argument should indicate the InputFile\n");
		printf("2nd Argument should indicate the OutputFileName\n");
		printf("3th argument should indicate the number of threads.(Optional) (1 default)\n");
		exit(1);
	}

	int num_threads = 1;

	if(argc == 4)
		num_threads = atoi(argv[3]);

	InputFileName = argv[1];
	OutputFileName = argv[2];

    	omp_set_num_threads(num_threads);

    	//Open the FileReader class
	std::string InputFilePath = InputFileName;

	//Read the Inputfile.
	FileReader Reader(InputFilePath.c_str());

	int v1,v2,Initial_Vertices,weight;;

	int nodes,edges;

	//firt line of the input file contains the number of nodes and edges
	Reader.get_nodes_edges(nodes,edges); 

	int count = 0;

	std::vector<std::vector<unsigned> > edge_lists;

	for(int i=0;i<edges;i++)
	{
		Reader.read_edge(v1,v2,weight);
		if(forward_order.find(v1) == forward_order.end())
			forward_order[v1] = count++;

		if(forward_order.find(v2) == forward_order.end())
			forward_order[v2] = count++;

		edge_lists.push_back(std::vector<unsigned>());
		edge_lists[i].push_back(v1);
		edge_lists[i].push_back(v2);
		edge_lists[i].push_back(weight);
	}

	FileWriter fout(OutputFileName.c_str(),forward_order.size(),edges);

	for(int i=0;i<edges;i++)
	{
		fout.write_edge(forward_order[edge_lists[i][0]],forward_order[edge_lists[i][1]],edge_lists[i][2]);
		edge_lists[i].clear();
	}

	FILE *file = fout.get_file();

	fprintf(file, "%d\n",0);
	fprintf(file, "%d\n",nodes);
	fprintf(file, "%d\n",forward_order.size());

	for(std::unordered_map<unsigned,unsigned>::iterator it = forward_order.begin(); it!=forward_order.end();
		it++)
	{
		fprintf(file,"%d %d\n",it->second,it->first);
	}

	fout.fileClose();

	edge_lists.clear();
	forward_order.clear();

	return 0;
}
