#ifndef DFS_HELPER_H
#define DFS_HELPER_H

struct dfs_helper
{
	const int NIL = -1;

	int Nodes;

	int *low;
	int *discovery;
	int *parent;

	std::stack<unsigned> _stack;

	unsigned char *status;

	dfs_helper(int N)
	{
		Nodes = N;

		low = new int[Nodes];
		discovery = new int[Nodes];
		parent = new int[Nodes];
		status = new unsigned char[Nodes];

		initialize_arrays();
	}

	void initialize_arrays()
	{
		for(int i=0; i<Nodes; i++)
		{
			low[i] = NIL;
			discovery[i] = NIL;
			parent[i] = NIL;
			status[i] = 0;
		}

		while(!_stack.empty())
			_stack.pop();
	}

	~dfs_helper()
	{
		delete[] low;
		delete[] discovery;
		delete[] parent;
		delete[] status;

		while(!_stack.empty())
			_stack.pop();
	}
};
	
#endif