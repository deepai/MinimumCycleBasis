#ifndef _FILE_READER_H
#define _FILE_READER_H

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include "utils.h"

extern "C" {
#include "mmio.h"
}

class FileReader {
	FILE *InputFileName;
	MM_typecode matcode;

	int ret_code;

	int M, N, Edges;

	inline void ERROR(std::string ch) {
		std::cerr << RED << ch << " " << RESET;
	}

public:

	FileReader(const char *InputFile) {

		if ((InputFileName = fopen(InputFile, "r")) == NULL) {
			ERROR(InputFile);
			exit(1);
		}

		if (mm_read_banner(InputFileName, &matcode) != 0) {
			ERROR("Could not process Matrix Market banner.\n");
			exit(1);
		}

		if (!(mm_is_matrix(matcode) && mm_is_coordinate(matcode)
				&& (mm_is_integer(matcode) || mm_is_real(matcode))
				&& (mm_is_symmetric(matcode) || mm_is_general(matcode)))) {
			ERROR("Sorry, this application does not support this mtx file. \n");
			exit(1);
		}

		if ((ret_code = mm_read_mtx_crd_size(InputFileName, &M, &N, &Edges))
				!= 0) {
			ERROR("Couldn't find all 3 parameters\n");
			exit(1);
		}
	}

	void get_nodes_edges(int &nodes, int &edges) {
		nodes = M;
		edges = Edges;
	}

	void read_edge(int &u, int &v, int &weight) {
		fscanf(InputFileName, "%d %d %d", &u, &v, &weight);
		u--;
		v--;
		// if(u >= v)
		// {
		// 	ERROR("u >= v\n");
		// 	exit(1);
		// }
	}

	FILE *get_file() {
		return InputFileName;
	}

	void fileClose() {
		fclose(InputFileName);
	}

};

#endif
