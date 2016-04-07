#ifndef __CYCLES_H
#define __CYCLES_H

struct cycle
{
	csr_tree *tree;
	unsigned non_tree_edge_index;
	int total_length;

	bool operator<(const cycle &rhs) const
	{
		return (total_length < rhs.total_length);
	}

	struct compare
	{
		bool operator()(cycle *lhs,cycle *rhs)
		{
			return (lhs->total_length < rhs->total_length);
		}
	};

	cycle(csr_tree *tr, unsigned index)
	{
		tree = tr;
		non_tree_edge_index = index;
	}

	unsigned get_root()
	{
		return tree->root;
	}
};

#endif