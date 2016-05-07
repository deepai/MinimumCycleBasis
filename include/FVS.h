#ifndef __FVS_H
#define __FVS_H

#include "CsrGraphMulti.h"

struct FVS
{
	csr_multi_graph *input_graph;

	double *W;
	int *degree;

	int Nodes;

	FVS(csr_multi_graph *graph);

	~FVS()
	{
		delete[] degree;
		delete[] W;
	}
};

#endif