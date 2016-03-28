#ifndef _CSR_GRAPH
#define _CSR_GRAPH 

#include <utility>
#include <algorithm>

#include <vector>
#include <string>

#include "FileWriter.h"

class csr_graph
{
	struct comparator
	{
		bool operator()(const std::pair<unsigned,unsigned> &a,const std::pair<unsigned,unsigned> &b) const
		{
			if(a.first < b.first)
				return true;
			else if(a.first > b.first)
				return false;
			else
				return (a.second < b.second);
		}
	};;

public:

	int Nodes;

	std::vector<unsigned> *rowOffsets;
	std::vector<unsigned> *columns;
	std::vector<unsigned> *rows;

	std::vector<unsigned> *degree;

	csr_graph()
	{
		rowOffsets = new std::vector<unsigned>();
		columns    = new std::vector<unsigned>();
		rows       = new std::vector<unsigned>();
		degree     = new std::vector<unsigned>();
	}


	void insert(int a,int b,bool direction)
	{
		columns->push_back(b);
		rows->push_back(a);

		if(!direction)
			insert(b,a,true);
	}

	~csr_graph()
	{
		rowOffsets->clear();
		columns->clear();
		rows->clear();
		degree->clear();
	}

	//Calculate the degree of the vertices and create the rowOffset
	void calculateDegreeandRowOffset()
	{
		rowOffsets->resize(Nodes + 1);
		degree->resize(Nodes);

		for(int i=0;i<Nodes;i++)
		{
			rowOffsets->at(i) = 0;
			degree->at(i) = 0;
		}

		rowOffsets->at(Nodes) = 0;

		//Allocate a pair array for rows and columns array
		std::vector<std::pair<unsigned,unsigned> > combined;
		
		//copy the elements from the row and column array
		for(int i=0;i<rows->size();i++)
			combined.push_back(std::make_pair(rows->at(i),columns->at(i)));

		//Sort the elements first by row, then by column
		std::sort(combined.begin(),combined.end(),comparator());

		//copy back the elements into row and columns
		for(int i=0;i<rows->size();i++)
		{
			rows->at(i) = combined[i].first;
			columns->at(i) = combined[i].second;
		}

		combined.clear();

		//Now calculate the row_offset

		for(int i=0;i<rows->size();i++)
		{
			unsigned curr_row = rows->at(i);
			
			rowOffsets->at(curr_row)++;
		}

		int prev = 0,current;

		for(int i=0;i<=Nodes;i++)
		{
			current = rowOffsets->at(i);
			rowOffsets->at(i) = prev;
			prev += current;

			if(i != Nodes )
				degree->at(i) = current - rowOffsets->at(i);
		}

		assert(rowOffsets->at(Nodes) == rows->size());

		
		#ifdef INFO
			printf("row_offset size = %d,columns size = %d\n",rowOffsets->size(),columns->size());
		#endif

	}

	//Print to a file.
	void PrintToFile(std::string &fileName,int global_node_count)
	{

		if(degree->size() == 0 )
			return;

		FileWriter Writer(fileName.c_str(),global_node_count,rows->size()/2);

		for(int i=0;i<rows->size();i++)
		{
			if(rows->at(i) > columns->at(i))
				Writer.write_edge(rows->at(i),columns->at(i));
		}

		Writer.fileClose();
	}


};

#endif