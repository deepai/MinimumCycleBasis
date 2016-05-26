#include "FVS.h"

#include <set>
#include <queue>
#include <climits>
#include "utils.h"
#include <cstring>

FVS::FVS(csr_multi_graph *graph) {
	input_graph = new csr_multi_graph();

	input_graph->copy(*graph);

	assert(input_graph->Nodes == graph->Nodes);
	assert(input_graph->rows->size() == graph->rows->size());

	Nodes = graph->Nodes;

	W = new double[input_graph->Nodes];

	node_status = new bool[input_graph->Nodes];
	edge_status = new bool[input_graph->rows->size()];
	is_vtx_in_fvs = new bool[input_graph->Nodes];

	for (int i = 0; i < Nodes; i++) {
		if (input_graph->degree->at(i) == 0)
			W[i] = 0;
		else
			W[i] = 1;

		is_vtx_in_fvs[i] = 0;
	}

	for (int i = 0; i < Nodes; i++)
		node_status[i] = 1;

	for (int i = 0; i < input_graph->rows->size(); i++)
		edge_status[i] = 1;
}

void FVS::pruning(int node_id) {
	std::set<int> elements_to_prune;

	elements_to_prune.insert(node_id);

	double C = W[node_id] / input_graph->degree->at(node_id);

	node_status[node_id] = 0;

	while (!elements_to_prune.empty()) {
		int first_element = *elements_to_prune.begin();

		if (node_status[first_element] == 0) {
			elements_to_prune.erase(first_element);
			continue;
		}

		//operations associate with removal of nodes.

		elements_to_prune.erase(first_element);
		input_graph->degree->at(first_element) = 0;
		W[first_element] = 0;

		int row, col;

		for (int i = input_graph->rowOffsets->at(first_element);
				i < input_graph->rowOffsets->at(first_element + 1); i++) {
			row = input_graph->rows->at(i);
			col = input_graph->columns->at(i);

			if (node_status[col] == 1) {
				input_graph->degree->at(col)--;if
(				input_graph->degree->at(col) <= 1)
				{
					elements_to_prune.insert(col);
				}
			}

			if (edge_status[i] != false) {
				edge_status[i] = false;
				edge_status[input_graph->reverse_edge->at(i)] = false;

				W[row] -= C;
				W[col] -= C;
			}
		}
	}
}

bool FVS::contains_cycle(int node_id, bool *visited, int *parent) {
	std::queue<int> bfs_queue;

	bfs_queue.push(node_id);

	bool global_break = false;

	visited[node_id] = 1;

	std::vector<unsigned char> edge_status(input_graph->rows->size());

	while (!bfs_queue.empty() && !global_break) {
		int nid = bfs_queue.front();

		assert(is_vtx_in_fvs[nid] == false);

		bfs_queue.pop();

		int col;

		for (int i = input_graph->rowOffsets->at(nid);
				i < input_graph->rowOffsets->at(nid + 1); i++) {
			col = input_graph->columns->at(i);
			if (is_vtx_in_fvs[col] == 1)
				continue;
			else if (!visited[col]) {
				bfs_queue.push(col);
				parent[col] = nid;
				visited[col] = true;
				edge_status[i] = 1;
				edge_status[input_graph->reverse_edge->at(i)] = 1;
			} else if (visited[col]) {
				if (col == parent[nid]) {
					if (edge_status[input_graph->reverse_edge->at(i)] == 1) {
						edge_status[i] = 1;
						continue;
					} else {
						global_break = true;
						break;
					}
				} else {
					global_break = true;
					break;
				}
			}
		}
	}
	while (!bfs_queue.empty())
		bfs_queue.pop();

	edge_status.clear();

	return global_break;
}

bool FVS::test_fvs() {
	bool *visited = new bool[Nodes];
	int *parent = new int[Nodes];

	bool found_cycle = false;

	for (int i = 0; i < Nodes; i++) {
		visited[i] = false;
		parent[i] = -1;
	}
	for (int i = 0; i < Nodes && !found_cycle; i++)
		if (!visited[i] && !is_vtx_in_fvs[i]) {
			found_cycle = contains_cycle(i, visited, parent);
		}

	delete[] visited;
	delete[] parent;

	return found_cycle;
}

void FVS::MGA() {
	int initial_zero_nodes = 0;

	for (int i = 0; i < Nodes; i++)
		initial_zero_nodes += (1 - node_status[i]);

	while (Nodes - initial_zero_nodes > 0) {

		double MAX_VAL = INT_MAX, temp_ratio;
		int vtx_min_ratio = 0;

		for (int i = 0; i < Nodes; i++)
			if (node_status[i]) {
				temp_ratio = W[i] / input_graph->degree->at(i);
				if (temp_ratio < MAX_VAL) {
					MAX_VAL = temp_ratio;
					vtx_min_ratio = i;
				}
			}

		FVS_SET.push_back(vtx_min_ratio);
		is_vtx_in_fvs[vtx_min_ratio] = true;

		initial_zero_nodes++;

		pruning(vtx_min_ratio);
	}

	FVS_SET.reverse();

	int node_id;

	//if the test contains any cycle by excluding {F/vi} vertex, then remove vi from F.
	for (std::list<int>::iterator it = FVS_SET.begin(); it != FVS_SET.end();) {

		node_id = *it;
		is_vtx_in_fvs[node_id] = false;

		bool val = !test_fvs();

		if (val) {
			FVS_SET.erase(it++);
			continue;
		} else {
			is_vtx_in_fvs[node_id] = true;
			it++;
		}
	}
}

int *FVS::get_copy_fvs_array() {
	int *fvs_output_array = new int[input_graph->Nodes];

	int count = 0;
	for (int i = 0; i < input_graph->Nodes; i++)
		if (is_vtx_in_fvs[i])
			fvs_output_array[i] = count++;
		else
			fvs_output_array[i] = -1;

	return fvs_output_array;
}

void FVS::print_fvs() {
#ifdef PRINT

	// printf("Number of FVS elements = %d\n",FVS_SET.size());
	// for(std::list<int>::iterator it = FVS_SET.begin(); it != FVS_SET.end();it++)
	// 	printf("%d ",*it + 1);
	// printf("\n");

#endif
}

int FVS::get_num_elements() {
	return FVS_SET.size();
}
