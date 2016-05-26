#ifndef _H_ISOMETRIC_CYCLE
#define _H_ISOMETRIC_CYCLE

#include "cycle_searcher.h"
#include "cycles.h"
#include "CsrTree.h"

struct isometric_cycle {
	int num_cycles;
	int *root;

	cycle_storage *storage;
	std::vector<cycle*> *list_cycles;

	isometric_cycle(int N, cycle_storage *store, std::vector<cycle*> *list) {
		num_cycles = N;
		root = new int[num_cycles];

		for (int i = 0; i < num_cycles; i++)
			root[i] = i;

		storage = store;
		list_cycles = list;
	}

	~isometric_cycle() {
		delete[] root;
	}

	void merge(int i, int j) {
		int root_x, root_y;

		root_x = find(i);
		root_y = find(j);

		if (root_x == root_y)
			return;

		if (list_cycles->at(root_x)->ID < list_cycles->at(root_y)->ID)
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

	/*
	 * This method is used to obtain the isometric cycles and only keep one cycle among the same list of isometric cycles.
	 *
	 */
	void obtain_isometric_cycles() {
		for (int i = 0; i < num_cycles; i++) {
			cycle *cle = list_cycles->at(i);

			assert(storage->list_trees[cle->tree->root] != NULL);
			assert(cle->ID < num_cycles);

			std::vector<unsigned> *s_values = cle->tree->s_values;

			unsigned row, col, src;

			row = cle->tree->parent_graph->rows->at(cle->non_tree_edge_index);
			col = cle->tree->parent_graph->columns->at(
					cle->non_tree_edge_index);

			src = cle->tree->root;

			if (s_values->at(row) != s_values->at(col)) {
				if ((src == row)) {
					cycle *match_cycle = storage->get_cycle(col, row, col, cle,
							cle->non_tree_edge_index);
					if (match_cycle != NULL) {
						merge(cle->ID, match_cycle->ID);

#ifdef PRINT
						printf("\n");
						cle->print_line();
						match_cycle->print_line();
						printf("\n");
#endif

					}
				} else {
					int r1 = s_values->at(row);
					std::vector<unsigned> *s_value_r1 = storage->get_s_value(
							r1);
					std::vector<unsigned> *s_value_col = storage->get_s_value(
							col);

					if ((s_value_r1 != NULL) && (src == s_value_r1->at(col))) {
						cycle *match_cycle = storage->get_cycle(r1, row, col,
								cle, cle->non_tree_edge_index);
						if (match_cycle != NULL) {
							merge(cle->ID, match_cycle->ID);

#ifdef PRINT
							printf("\n");
							cle->print_line();
							match_cycle->print_line();
							printf("\n");
#endif
						}
					} else if ((s_value_col != NULL)
							&& (row == s_value_col->at(r1))) {
						cycle *match_cycle = storage->get_cycle(col, src, r1,
								cle);
						if (match_cycle != NULL) {
							merge(cle->ID, match_cycle->ID);

#ifdef PRINT
							printf("\n");
							cle->print_line();
							match_cycle->print_line();
							printf("\n");
#endif
						}
					}
				}
			} else {
			}
		}

		for (int i = 0; i < num_cycles; i++) {
			if (list_cycles->at(i) == NULL)
				continue;
			if (root[list_cycles->at(i)->ID] != list_cycles->at(i)->ID) {
				delete list_cycles->at(i);
				list_cycles->at(i) = NULL;
			}
		}
	}

};

#endif
