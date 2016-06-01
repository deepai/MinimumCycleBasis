#ifndef _FILE_WRITER_H
#define _FILE_WRITER_H

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include "utils.h"

extern "C" {
#include "mmio.h"
}

class FileWriter {
	FILE *OutputFileName;
	MM_typecode matcode;

	int ret_code;

	int M, N, Edges;

	inline void ERROR(const char *ch) {
		std::cerr << RED << ch << " " << RESET;
	}

public:

	FileWriter(const char *OutputFile, int Nodes, int NZ) {
		M = Nodes;
		Edges = NZ;

		if ((OutputFileName = fopen(OutputFile, "w")) == NULL) {
			ERROR("Unable to open file.\n");
			printf("filename = %s\n", OutputFile);
			exit(1);
		}

		mm_initialize_typecode(&matcode);
		mm_set_matrix(&matcode);
		mm_set_coordinate(&matcode);
		mm_set_integer(&matcode);
		mm_set_symmetric(&matcode);

		mm_write_banner(OutputFileName, matcode);
		mm_write_mtx_crd_size(OutputFileName, Nodes, Nodes, Edges);
	}

	void write_edge(int u, int v, int weight) {
		fprintf(OutputFileName, "%d %d %d\n", u + 1, v + 1, weight);
	}

	FILE *get_file() {
		return OutputFileName;
	}

	void fileClose() {
		fclose(OutputFileName);
	}

};

#endif
