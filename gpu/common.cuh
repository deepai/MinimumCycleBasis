#ifndef __H_COMMON_CUDA
#define __H_COMMON_CUDA

#define calculate_8bit(X) (X)
#define calculate_16bit(X) (2*X)
#define calculate_32bit(X) (4*X)
#define calculate_64bit(X) (8*X)

void func(int a,int b);

#endif