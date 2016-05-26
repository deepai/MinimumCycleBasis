#ifndef __UTIL_H
#define __UTIL_H

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <assert.h>

#ifdef __NVCC__

#include <cuda_runtime.h>
#include <cuda.h>

#endif
//the following are UBUNTU/LINUX ONLY terminal color codes.
#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */

#ifdef __NVCC__

#define CudaError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort =
		true) {
	if (code != cudaSuccess) {
		std::cerr << RED << "Error :" << cudaGetErrorString(code) << " : "
		<< file << " : line No = " << line << RESET << std::endl;
		// fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
		cudaDeviceReset();
		exit(code);
	}
}

#endif

struct debugger {
	template<typename T> debugger& operator ,(const T& v) {
		std::cerr << CYAN << v << " " << RESET;
		return *this;
	}
};

#define BLOCK_DEFAULT 1024
#define CEIL(SIZE) ((int)ceil(((double)SIZE)/BLOCK_DEFAULT))

typedef unsigned long long unsignedLL;

extern debugger dbg;

#ifdef VERBOSE
#define debug(args...)            {dbg,args; std::cerr<<std::endl;}
#else
#define debug(args...)            {}
#endif 

#endif
