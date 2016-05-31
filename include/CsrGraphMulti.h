#ifndef __CSR_MULTI_GRAPH
#define __CSR_MULTI_GRAPH

#include "CsrGraph.h"
#include <unordered_map>
#include <map>

class csr_multi_graph: public csr_graph {

protected:
	struct edge {
		unsigned row;
		unsigned col;
		int weight;
		int chain_index;

		struct edge *reverse_edge_ptr;
		int original_edge_index;
		int reverse_edge_index;

		edge(unsigned r, unsigned c, int w, int ch_index, int orig_index) {
			row = r;
			col = c;
			weight = w;
			chain_index = ch_index;
			original_edge_index = orig_index;
		}
	};

	struct compare {
		bool operator()(const edge *a, const edge *b) const {
			if (a->row == b->row)
				return (a->col < b->col);
			else
				return (a->row < b->row);
		}
	};
public:
	std::vector<unsigned> *reverse_edge;
	std::vector<int> *chains;
	std::vector<int> *edge_original_graph;
	csr_multi_graph() {
		reverse_edge = new std::vector<unsigned>();
		chains = new std::vector<int>();
		edge_original_graph = new std::vector<int>();
	}

	~csr_multi_graph() {
		reverse_edge->clear();
		chains->clear();
		edge_original_graph->clear();
	}

	void copy(const csr_multi_graph& other) {
		this->Nodes = other.Nodes;
		this->initial_edge_count = other.initial_edge_count;

		for (int i = 0; i < other.rows->size(); i++) {
			this->rows->push_back(other.rows->at(i));
			this->columns->push_back(other.columns->at(i));
			this->weights->push_back(other.weights->at(i));
			this->reverse_edge->push_back(other.reverse_edge->at(i));
		}

		for (int i = 0; i < Nodes; i++) {
			this->degree->push_back(other.degree->at(i));
		}

		for (int i = 0; i < other.rowOffsets->size(); i++)
			this->rowOffsets->push_back(other.rowOffsets->at(i));

		return;
	}

	void insert(int a, int b, int wt, int chain_index, int edge_index,
			bool direction) {
		columns->push_back(b);
		rows->push_back(a);
		weights->push_back(wt);
		chains->push_back(chain_index);
		edge_original_graph->push_back(edge_index);

		if (!direction)
			reverse_edge->push_back(rows->size());
		else
			reverse_edge->push_back(rows->size() - 2);

		if (!direction)
			insert(b, a, wt, chain_index, edge_index, true);
	}
	//Calculate the degree of the vertices and create the rowOffset
	/**
	 * @brief [brief description]
	 * @details [long description]
	 */
	void calculateDegreeandRowOffset() {
		rowOffsets->resize(Nodes + 1);
		degree->resize(Nodes);

		for (int i = 0; i < Nodes; i++) {
			rowOffsets->at(i) = 0;
			degree->at(i) = 0;
		}

		rowOffsets->at(Nodes) = 0;

		//Allocate a pair array for rows and columns array
		std::vector<edge*> combined;

		//copy the elements from the row and column array
		for (int i = 0; i < rows->size(); i++)
			combined.push_back(
					new edge(rows->at(i), columns->at(i), weights->at(i),
							chains->at(i), edge_original_graph->at(i)));

		//assing the reverse_edge_pointers to the correct edge pointers.
		for (int i = 0; i < rows->size(); i++)
			combined[i]->reverse_edge_ptr = combined[reverse_edge->at(i)];

		//Sort the elements first by row, then by column
		std::sort(combined.begin(), combined.end(), compare());

		for (int i = 0; i < rows->size(); i++)
			combined[i]->reverse_edge_ptr->reverse_edge_index = i;

		//copy back the elements into row and columns
		for (int i = 0; i < rows->size(); i++) {
			rows->at(i) = combined[i]->row;
			columns->at(i) = combined[i]->col;
			weights->at(i) = combined[i]->weight;
			chains->at(i) = combined[i]->chain_index;
			edge_original_graph->at(i) = combined[i]->original_edge_index;

			assert(combined[i]->reverse_edge_index < rows->size());

			reverse_edge->at(i) = combined[i]->reverse_edge_index;
		}

		for (int i = 0; i < rows->size(); i++)
			delete combined[i];

		combined.clear();
		//Now calculate the row_offset

		for (int i = 0; i < rows->size(); i++) {
			unsigned curr_row = rows->at(i);

			rowOffsets->at(curr_row)++;}

		unsigned prev = 0, current;

		for (int i = 0; i <= Nodes; i++) {
			current = rowOffsets->at(i);
			rowOffsets->at(i) = prev;
			prev += current;
		}

		for (int i = 0; i < Nodes; i++) {
			degree->at(i) = rowOffsets->at(i + 1) - rowOffsets->at(i);
		}

		assert(rowOffsets->at(Nodes) == rows->size());

#ifdef INFO
		printf("row_offset size = %d,columns size = %d\n",rowOffsets->size(),columns->size());
#endif

	}

	/**
	 * @brief [brief description]
	 * @details [long description]
	 *
	 * @param r [description]
	 * @param c [description]
	 * @param e [description]
	 * @param tree_edges [description]
	 * @param src [description]
	 */
	void fill_tree_edges(unsigned *r, unsigned *c, int *e,
			std::vector<unsigned> *tree_edges, unsigned src) {
		assert(tree_edges->size() + 1 == Nodes);

		std::vector<edge*> temporary_array;
		unsigned row, col;
		for (int i = 0; i < tree_edges->size(); i++) {
			row = rows->at(tree_edges->at(i));
			col = columns->at(tree_edges->at(i));

			temporary_array.push_back(
					new edge(row, col, 0, 0, (int) tree_edges->at(i)));
		}
		sort(temporary_array.begin(), temporary_array.end(), compare());

		for (int i = 0; i < temporary_array.size(); i++) {
			r[temporary_array[i]->row]++;
			c[i] = temporary_array[i]->col;
			e[i] = temporary_array[i]->original_edge_index;
		}

		e[temporary_array.size()] = -1;

		unsigned prev = 0, current;

		for (int i = 0; i <= Nodes; i++) {
			current = r[i];
			r[i] = prev;
			prev += current;
		}

		for (int i = 0; i < temporary_array.size(); i++)
			delete temporary_array[i];

		temporary_array.clear();
	}

	/**
	 * @brief [brief description]
	 * @details [long description]
	 *
	 * @param non_tree_edges [description]
	 * @param src [description]
	 *
	 * @return [description]
	 */
	std::vector<unsigned> *get_spanning_tree(
			std::vector<unsigned> **non_tree_edges, int src);

	static csr_multi_graph *get_modified_graph(csr_graph *graph,
			std::vector<unsigned> *remove_edge_list,
			std::vector<std::vector<unsigned> > *edges_new_list,
			int nodes_removed) {
		std::vector<bool> filter_edges(graph->rows->size());
		for (int i = 0; i < filter_edges.size(); i++)
			filter_edges[i] = false;

		for (int i = 0;
				(remove_edge_list != NULL) && (i < remove_edge_list->size());
				i++)
			filter_edges[remove_edge_list->at(i)] = true;

		csr_multi_graph *new_reduced_graph = new csr_multi_graph();

		std::unordered_map<unsigned, unsigned> *new_nodes =
				new std::unordered_map<unsigned, unsigned>();

		int new_node_count = 0;
		//This is for Relabelling vertices.
		for (int i = 0; i < graph->rows->size(); i++) {
			if (!filter_edges.at(i)) {
				if (new_nodes->find(graph->rows->at(i)) == new_nodes->end())
					new_nodes->insert(
							std::make_pair(graph->rows->at(i),
									new_node_count++));
				if (new_nodes->find(graph->columns->at(i)) == new_nodes->end())
					new_nodes->insert(
							std::make_pair(graph->columns->at(i),
									new_node_count++));
			}
		}

		for (int i = 0;
				(edges_new_list != NULL) && (i < edges_new_list->size()); i++) {
			if (new_nodes->find(edges_new_list->at(i)[0]) == new_nodes->end())
				new_nodes->insert(
						std::make_pair(edges_new_list->at(i)[0],
								new_node_count++));
			if (new_nodes->find(edges_new_list->at(i)[1]) == new_nodes->end())
				new_nodes->insert(
						std::make_pair(edges_new_list->at(i)[1],
								new_node_count++));
		}

		new_reduced_graph->Nodes = new_node_count;
		//We have the relabel information now and can easily fill the edges.
		//add new edges first.
		for (int i = 0;
				(edges_new_list != NULL) && (i < edges_new_list->size()); i++) {
			new_reduced_graph->insert(new_nodes->at(edges_new_list->at(i)[0]),
					new_nodes->at(edges_new_list->at(i)[1]),
					edges_new_list->at(i)[2], i, -1, false);
		}
		//add the old edges
		for (int i = 0; i < graph->rows->size(); i++) {
			if (!filter_edges.at(i)) {
				if (graph->rows->at(i) < graph->columns->at(i))
					new_reduced_graph->insert(new_nodes->at(graph->rows->at(i)),
							new_nodes->at(graph->columns->at(i)),
							graph->weights->at(i), -1, i, false);
			}
		}

		new_nodes->clear();

		new_reduced_graph->calculateDegreeandRowOffset();

		filter_edges.clear();

		return new_reduced_graph;
	}
};

#endif
