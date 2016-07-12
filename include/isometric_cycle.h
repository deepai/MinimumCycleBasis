#ifndef _H_ISOMETRIC_CYCLE
#define _H_ISOMETRIC_CYCLE

#include "cycle_searcher.h"
#include "cycles.h"
#include "CsrTree.h"

struct isometric_cycle {
	int num_cycles;
	int *root;

	cycle_storage *storage;
	std::vector<cycle*> list_cycles;

	isometric_cycle(int N, cycle_storage *store) {
		num_cycles = N;
		root = new int[num_cycles];

		for (int i = 0; i < num_cycles; i++)
			root[i] = i;

		storage = store;
		list_cycles.resize(N);

		int i = 0;

		for(int j = 0; j < storage->list_cycles.size(); j++){
		 	for(auto pair : storage->list_cycles[j]){
		 		for(auto cle : pair.second){
		 			cle->ID = i;
		 			list_cycles[i++] = cle;
		 		}
		 	}
		}

		assert(i == num_cycles);
	}

	~isometric_cycle() {
		delete[] root;
		list_cycles.clear();
	}

	void merge(int i, int j) {
		int root_x, root_y;

		root_x = find(i);
		root_y = find(j);

		if (root_x == root_y)
			return;

		if (list_cycles[root_x]->ID < list_cycles[root_y]->ID)
			root[root_y] = root_x;
		else
			root[root_x] = root_y;
	}

	int find(int i) {
		if (root[i] == i)
			return i;
		else
			root[i] = find(root[i]);
		return root[i];
	}

// 	/*
// 	 * This method is used to obtain the isometric cycles and only keep one cycle among the same list of isometric cycles.
// 	 *
// 	 */
	void obtain_isometric_cycles()
	{
		for (int i = 0; i < num_cycles; i++) {
 			cycle *cle = list_cycles[i];

 //			assert(storage->list_trees[cle->tree->root] != NULL);
 			assert(cle->ID < num_cycles);

// 			std::vector<unsigned> *s_values = cle->tree->s_values;

 			unsigned row, col, src;

 			row = cle->trees->parent_graph->rows->at(cle->non_tree_edge_index);
 			col = cle->trees->parent_graph->columns->at(cle->non_tree_edge_index);

 			src = cle->root;

 			int reverse_edge = cle->trees->parent_graph->reverse_edge->at(cle->non_tree_edge_index);

 			if (cle->Su != cle->Sv) {
				if ((src == row)) {
 					std::vector<cycle*> *match_cycles = storage->match_cycles(src,cle->non_tree_edge_index);
 					if (match_cycles != NULL) {
 						for(int j=0;j<match_cycles->size();j++)
	 						merge(cle->ID, match_cycles->at(j)->ID);
 					}
 				}
 				else
 				{
 					int r1 = cle->Su;
 					std::vector<cycle*> *match_cycles = storage->match_cycles(r1,cle->non_tree_edge_index);
 					if (match_cycles != NULL) {
 						for(int j=0;j<match_cycles->size();j++)
 						{
 							int edge = match_cycles->at(j)->non_tree_edge_index;

 							int u_temp = cle->trees->parent_graph->rows->at(edge);
 							int v_temp = cle->trees->parent_graph->columns->at(edge);

 							if(u_temp == row && v_temp == col){
 							 	if(match_cycles->at(j)->Sv == src){
 									merge(cle->ID,match_cycles->at(j)->ID);
 								}
 								else
 								{
 									int r_edge = cle->trees->parent_graph->reverse_edge->at(cle->Ou);
 									std::vector<cycle*> *match_next_cycles = storage->match_cycles(col,cle->Ou);
 									for(int k=0;k<match_next_cycles->size();k++){

 										int edge_temp = match_next_cycles->at(k)->non_tree_edge_index;
 										int u_temp2 = cle->trees->parent_graph->rows->at(edge_temp);
 										int v_temp2 = cle->trees->parent_graph->columns->at(edge_temp);

 										if(u_temp2 == r1 && v_temp2 == src)
 										{
 											if(row == match_next_cycles->at(k)->Su)
 											{
 												merge(cle->ID,match_next_cycles->at(k)->ID);
 											}
 										}
 									}
 								}
 							}
 						}
 					}
 				}
 			}
 		}
 	}

 	int count_isometric_cycles()
 	{
 		int count = 0;
 		for(int i=0;i<num_cycles;i++)
 			if(root[i] == i)
 				count++;
 		return count;
 	}

};

#endif
