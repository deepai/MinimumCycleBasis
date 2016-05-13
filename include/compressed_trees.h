#ifndef _H_COMPRESSED_GRAPH
#define _H_COMPRESSED_GRAPH

#include "CsrGraphMulti.h"
#include <cstring>

struct compressed_trees
{
	int num_rows;

	int fvs_size;

	int original_nodes;

	int chunk_size;

	unsigned **tree_rows; //rowoffsets
	unsigned **tree_cols; //columns

	int **edge_offset; //offset of the corresponding edge in the csr format
	int **parent;      //parent array corresponding to each edge

	int **distance;    //distance array in terms of unweighted edge.

	unsigned **precompute_value;  //This is used to store the precomputed value corresponding to each tree.

	csr_multi_graph *parent_graph;

	unsigned *final_vertices;  //contains the final fvs vertices.
	int *vertices_map;    //contains the index of the fvs vertices and -1 if the vertex doesn't belong to fvs.

	compressed_trees(int chunk,int N,int *fvs_array,csr_multi_graph *graph)
	{
		fvs_size = N;
		chunk_size = chunk;
		parent_graph = graph;
		original_nodes = graph->Nodes;

		int r = (int)ceil((double)N/chunk_size);

		num_rows = r;

		tree_rows = new unsigned*[num_rows];
		tree_cols = new unsigned*[num_rows];
		edge_offset = new int*[num_rows];
		parent = new int*[num_rows];
		distance = new int*[num_rows];

		precompute_value = new unsigned*[num_rows];

		for(int i=0;i<num_rows;i++)
		{
			tree_rows[i] = new unsigned[chunk * (original_nodes + 1)];
			tree_cols[i] = new unsigned[chunk * original_nodes];
			edge_offset[i] = new int[chunk * original_nodes];
			parent[i] = new int[chunk * original_nodes];
			distance[i] = new int[chunk * original_nodes];

			precompute_value[i] = new unsigned[chunk * original_nodes];

			memset(tree_rows[i],0,sizeof(unsigned) * chunk * (original_nodes + 1));
		}

		final_vertices = new unsigned[fvs_size];

		vertices_map = fvs_array;

		for(int i=0;i<original_nodes;i++)
			if(vertices_map[i] != -1)
			{
				assert(vertices_map[i] < fvs_size);
				final_vertices[vertices_map[i]] = i;
			}

	}

	~compressed_trees()
	{
		for(int i=0;i<num_rows;i++)
		{
			delete[] tree_rows[i];
			delete[] tree_cols[i];
			delete[] parent[i];
			delete[] edge_offset[i];
			delete[] precompute_value[i];
			delete[] distance[i];
		}

		delete [] tree_rows;
		delete [] tree_cols;
		delete [] parent;
		delete [] edge_offset;
		delete [] precompute_value;
		delete [] distance;
		delete [] final_vertices;
	}

	int get_node_arrays(unsigned **csr_rows,unsigned **csr_cols,int **csr_edge_offset,int **csr_parent,int **csr_distance,int node_index);

	int get_precompute_array(unsigned **precompute_tree,int node_index);

	int get_index(int original_node);

	void compressed_trees::copy(int index,std::vector<unsigned> *tree_edges,
							std::vector<int> *parent_edges,std::vector<int> *distances);
};

#endif