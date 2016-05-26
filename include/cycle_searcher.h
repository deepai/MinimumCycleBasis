#ifndef _H_CYCLE_SEARCH
#define _H_CYCLE_SEARCH

#include <unordered_map>
#include <vector>
#include <utility>
#include <climits>

#include "cycles.h"

struct list_common_cycles {
	std::vector<cycle*> listed_cycles;

	list_common_cycles(cycle *cle) {
		listed_cycles.push_back(cle);
	}

	inline void add_cycle(cycle *cle) {
		listed_cycles.push_back(cle);
	}
};

struct cycle_storage {
	int Nodes;
	std::vector<std::unordered_map<unsigned long long, list_common_cycles*> > list_cycles;

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

	void add_cycle(unsigned root, unsigned u, unsigned v, cycle *cle) {
		unsigned long long index = combine(std::min(u, v), std::max(u, v));

		if (list_cycles[root].find(index) == list_cycles[root].end())
			list_cycles[root].insert(
					std::make_pair(index, new list_common_cycles(cle)));
		else
			list_cycles[root][index]->add_cycle(cle);
	}

	void clear_cycles() {
		for (int i = 0; i < list_cycles.size(); i++) {
			list_cycles[i].clear();
		}
		list_cycles.clear();
	}
};

#endif
