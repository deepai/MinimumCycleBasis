#ifndef BICC_H
#define BICC_H

#include <map>
#include <vector>
#include <utility>
#include <set>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <list>

#include "FileWriter.h"
#include <fstream>
#include "Host_Timer.h"
#include "CsrGraph.h"

#include <assert.h>

//Struct BCC is used to contain information about Biconnected Components
//Contains Nodes, Edges and adjacencyList std::map<int,std::set<int>* > 
struct bicc_graph
{
	int Nodes,Edges;
	
	csr_graph *c_graph;

	std::vector<int> bicc_number;

	bicc_graph(int _Nodes)
	{
		Nodes = _Nodes;
		Edges = 0;
		c_graph = new csr_graph();
	}

	~bicc_graph()
	{
		delete c_graph;
	}

	//Insert edge between i and j. If direction == 0 , Insert i->j and j->i , 
	//else if direction == 1 , insert only i->j
	void insert_edge(int i,int j,bool direction)
	{
		c_graph->insert(i,j,direction);
	}

	void calculate_nodes_edges()
	{
		c_graph->Nodes = Nodes;

		c_graph->calculateDegreeandRowOffset();
		Edges = c_graph->rows->size();
	}

	inline unsigned long long merge(unsigned long long upper,unsigned long long lower)
	{
		unsigned long long result = 0;
		result = ((upper << 32) | lower);
		return result;
	}

	/**
	 * @brief This method is used to remove nodes having degree less than a given threshold.
	 * @details We first calculate the degree of nodes present in the current component.
	 * The degree of each node of the current component is calculated from the given edge_list.
	 * We search in the adjacent list of nodes which has degree less than the given threshold 
	 * and subtract the degree of the adjacent nodes. We make the degree of the current node as 0.
	 * This Method is Thread_Safe and can be used with OMP.
	 * 
	 * @param degree_threshold threshold.
	 * @param component_number Component Number.
	 * @param edge_list edge_list 
	 * 
	 * @return count of edges pruned;
	 */
	int prune_edges(int degree_threshold,int component_number,std::list<int> *edge_list,
		std::unordered_map<unsigned long long,int> *edge_map,
		std::unordered_map<int,int> &src_vtx_component)
	{
		int count_edges_pruned = 0;

		std::unordered_map<unsigned,int> degree_nodes;

		unsigned src_vtx;
		unsigned dest_vtx;

		for(std::list<int>::iterator it=edge_list->begin(); it != edge_list->end(); it++)
		{
			src_vtx = c_graph->rows->at(*it);
			dest_vtx = c_graph->columns->at(*it);

			////debug("edges for component,",component_number,"are",src_vtx + 1,dest_vtx + 1);
			
			degree_nodes[src_vtx]++;
			degree_nodes[dest_vtx]++;
		}

		// for(std::unordered_map<unsigned,int>::iterator it = degree_nodes.begin();
		// 	it!=degree_nodes.end();it++)
		// {
		// 	////debug("degree",it->first + 1,it->second);
		// 	//it->second /= 2;
		// }

		bool all_vertices_pruned = false;

		while(!all_vertices_pruned)
		{
			all_vertices_pruned = true;

			for(std::unordered_map<unsigned,int>::iterator it= degree_nodes.begin();
				it!=degree_nodes.end(); it++)
			{
				if( (it->second <= degree_threshold)  && (it->second > 0) )
				{
					////debug("vertex:",it->first,"degree_threshold:",degree_threshold);

					all_vertices_pruned = false;

					src_vtx = it->first;
					for(int j=c_graph->rowOffsets->at(src_vtx); j<c_graph->rowOffsets->at(src_vtx + 1); 
						j++)
					{
						dest_vtx = c_graph->columns->at(j);

						if(bicc_number[j] != component_number)
							continue;

						degree_nodes[dest_vtx]--;

						if(degree_nodes[dest_vtx] > degree_threshold)
							src_vtx_component[component_number] = dest_vtx;


						int reverse_edge = edge_map->at(merge(dest_vtx,src_vtx));

						bicc_number[reverse_edge] = -1;
						bicc_number[j] = -1;

						//debug("removed",src_vtx + 1,dest_vtx + 1);
						//debug("removed",dest_vtx + 1,src_vtx + 1);

						count_edges_pruned+=2;
					}

					degree_nodes[src_vtx] = 0;
				}
			}
		}

		degree_nodes.clear();
		return count_edges_pruned++;
	}


 	/**
 	 * @brief This method is used to collect edges for each component and one source vertex
 	 * per component.
 	 * @details Traverse through the entire edge lists and collect only the vertices present in the new range of
 	 * component numbers.
 	 * 
 	 * @param component_range_start start range inclusive of component numbers.
 	 * @param component_range_end end range inclusive of component numbers
 	 * @param src_vtx_component map of component number => src vertex
 	 */
	void collect_edges_component(int component_range_start,int component_range_end,
		std::unordered_map<int,std::list<int>* > &edge_list_component,
		std::unordered_map<int,int> &src_vtx_component)
	{
		for(int j=0; j<Edges; j++)
		{
			if((bicc_number[j]>=component_range_start) && (bicc_number[j]<=component_range_end))
			{
				if(edge_list_component.find(bicc_number[j]) == edge_list_component.end())
				{
					std::list<int> *temp=new std::list<int>();
					temp->push_back(j);

					edge_list_component[bicc_number[j]] = temp;
				}
				else
					edge_list_component[bicc_number[j]]->push_back(j);

				//add to src_vtx_component
				src_vtx_component[bicc_number[j]] = c_graph->columns->at(j);
			}
		}

		for(std::unordered_map<int,std::list<int>* >::iterator it = edge_list_component.begin();
			it != edge_list_component.end(); it++)
		{
			for(std::list<int>::iterator ij = it->second->begin();ij != it->second->end();
				ij++)
			{
				int edge_index = *ij;
				////debug(it->first,c_graph->rows->at(edge_index) + 1,c_graph->columns->at(edge_index) + 1);
			}
		}

	}

	/**
	 * @brief Initially all edges of the graph belong to the same bicc. Hence bicc_number is initialized 
	 * to 1.
	 * @details [long description]
	 * 
	 */
	void initialize_bicc_numbers()
	{
		assert(c_graph->rowOffsets->at(Nodes) == Edges);

		bicc_number.resize(Edges);
		for(int i=0; i<Edges ;i++)
			bicc_number[i] = 1;
	}

	/**
	 * @brief This method is used to print the active bccs to a file.
	 * @details Traverse the edge lists and store the edges in a <int,vector> pair corresponding 
	 * to each component number/
	 * 
	 * @param ID 
	 * @param outputDirName Output Directory
	 * @param global_nodes_count Original Node Count of each file.
	 * @param start_range Start Range of biconnected component
	 * @param end_range End Range of biconnected components.
	 * @return [description]
	 */
	double print_to_a_file(int &file_output_count,std::string outputDirName,int global_nodes_count,
		std::unordered_set<int> &finished_components)
	{
		std::string statsFileName = outputDirName + "stats";
		//This is used to gather the edge list corresponding to each component in the graph.
		std::unordered_map<int,std::list<int> > edge_list;
		std::unordered_map<int,std::unordered_set<int> > count_nodes;

		for(int i=0;i<Edges;i++)
		{
			if( finished_components.find(bicc_number[i]) != finished_components.end() )
			{
				int src_vtx = c_graph->rows->at(i);
				int dest_vtx = c_graph->columns->at(i);

				if(count_nodes.find(bicc_number[i]) == count_nodes.end())
				{
					count_nodes[bicc_number[i]] = std::unordered_set<int>();
				}

				//insert bicc_numbers[i]
				count_nodes[bicc_number[i]].insert(src_vtx);
				count_nodes[bicc_number[i]].insert(dest_vtx);

				//print only the edge where src_vtx > dest_vtx;
				if(src_vtx <= dest_vtx)
					continue;

				if(edge_list.find(bicc_number[i]) != edge_list.end() )
					edge_list[bicc_number[i]].push_back(i);
				else
				{
					edge_list[bicc_number[i]] = std::list<int>();
					edge_list[bicc_number[i]].push_back(i);
				}
			}
		}

		for(std::unordered_map<int,std::list<int> >::iterator it=edge_list.begin();
			it != edge_list.end(); it++)
		{
			int component_number = it->first;

			if(it->second.size() > 1)
			{
				++file_output_count;

				std::string outputfilePath = outputDirName + std::to_string(file_output_count) + ".mtx";

				//debug(outputfilePath);

				FileWriter fout(outputfilePath.c_str(),global_nodes_count,it->second.size());

				for(std::list<int>::iterator ij = it->second.begin(); ij != it->second.end();
					ij++)
				{
					int edge_index = *ij;

					int src_vtx = c_graph->rows->at(edge_index);
					int dest_vtx = c_graph->columns->at(edge_index);

					fout.write_edge(src_vtx,dest_vtx);
				}

				fout.fileClose();

				//Entry into the stats file
				std::ofstream fstats ( statsFileName.c_str(),std::ios::out | std::ios::app);

				fstats << file_output_count << " " << count_nodes[component_number].size() << " " << it->second.size() << std::endl;

				fstats.close();

			}
		}

	}
};


#endif
