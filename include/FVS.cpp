#include "FVS.h"

FVS::FVS(csr_multi_graph *graph)
{
	input_graph = graph;

	Nodes = graph->Nodes;

	degree = new int[input_graph->Nodes];
	W = new double[input_graph->Nodes];

	for(int i=0;i<Nodes;i++)
	{
		degree[i] = graph->degree->at(i);
		if(degree[i] == 0)
			W[i] = 0;
		else
			W[i] = 1;
	}
}