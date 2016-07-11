#ifndef _H_CYCLE_SEARCH
#define _H_CYCLE_SEARCH

#include <unordered_map>
#include <vector>
#include <utility>
#include <climits>

#include "cycles.h"

struct cycle_storage {
	int Nodes;
	std::vector<std::unordered_map<int, std::vector<cycle*> > > list_cycles;

	inline unsigned long long combine(unsigned u, unsigned v) {
		unsigned long long value = u;
		value <<= 32;

		value = value | v;

		return value;
	}

	cycle_storage(int N) {
		Nodes = N;
		list_cycles.resize(Nodes);
	}

	~cycle_storage() {
		list_cycles.clear();
	}

	void add_cycle(unsigned root, int edge_index , cycle *cle) {
			list_cycles[root][edge_index].push_back(cle);
	}

	void clear_cycles() {
		for (int i = 0; i < list_cycles.size(); i++) {
			list_cycles[i].clear();
		}
		list_cycles.clear();
	}
};

#endif
