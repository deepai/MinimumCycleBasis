IDIR =../include
ODIR =obj
SRCDIR=src

EX_INCLUDE = ../include

#DFLAGS = -DMULTI_THREAD -DPRINT -DVERBOSE
DFLAGS = -DMULTI_THREAD  -DNDEBUG -DVERBOSE

CC=g++

CFLAGS= -O2 -I$(IDIR) -I$(EX_INCLUDE) -I../ -fopenmp -lgomp $(DFLAGS) -w -std=c++11 -fpermissive
CXXFLAGS = -O2 -I$(IDIR) -I$(EX_INCLUDE) -I../ -fopenmp -lgomp $(DFLAGS) -w -std=c++11 -fpermissive

NVFLAGS = -O2 -arch=sm_35 -Xcompiler -fopenmp -lgomp $(DFLAGS) -w -std=c++11 -Xptxas -v

NVCC = nvcc $(NVFLAGS) -I$(IDIR) -I$(EX_INCLUDE) -I../

#-DMULTI_THREAD for multi-thread support
#-DPRINT to print the bNVCCs in a file

LIBS=-lm

GPU_DIR = ../gpu
GPU_SRC = $(GPU_DIR)/*.cu
GPU_HEADERS = $(GPU_DIR)/*.cuh

_DEPS = *.h 
DEPS = $(patsubst %,$(EX_INCLUDE)/%,$(_DEPS))

_EXTRA_DEPS = mmio.o \
			  Host_Timer.o \
			  CsrGraphSpanningTree.o \
			  CsrGraphDegreeTwoRemoval.o \
			  CsrGraphMultiSpanningTree.o \
			  FVS.o \
			  compressed_trees.o
EXTRA_DEPS = $(patsubst %,$(EX_INCLUDE)/%,$(_EXTRA_DEPS))

GPU_DEPS = $(GPU_DIR)/gpu_struct.o \
		   $(GPU_DIR)/kernel_init_edges.o \
		   $(GPU_DIR)/common.o \
		   $(GPU_DIR)/kernel_multi_search.o

OBJ1 = mcb_cycle_deg_two_remove.o $(GPU_DEPS)
OBJ2 = mcb_cycle.o

all: mcb_cpu_baseline mcb_cuda Test_deg_two_remove Test mcb_cpu

$(GPU_DIR)/%.o: $(GPU_DIR)/%.cu $(GPU_HEADERS)
	@echo [NVCC] $@
	@$(NVCC) -dc -o $@ $< -I../

mcb_cuda: $(OBJ1) $(EXTRA_DEPS)
	@echo [NVCC] $@
	@$(NVCC) -o $@ $^ $(LIBS) $(DFLAGS)

mcb_cycle_deg_two_remove.o: mcb_cycle_deg_two_remove.cpp $(DEPS)
	@echo [NVCC] $@
	@$(NVCC) -c -o $@ $< $(DFLAGS) -I../

Test_deg_two_remove: Test_deg_two_remove.o $(EXTRA_DEPS)
	@echo [NVCC] $@
	@$(NVCC) -o $@ $^ $(LIBS) $(DFLAGS)

Test_deg_two_remove.o: Test_deg_two_remove.cpp $(DEPS)
	@echo [NVCC] $@
	@$(NVCC) -c -o $@ $< $(DFLAGS)

mcb_cpu_baseline: $(OBJ2) $(EXTRA_DEPS)
	@echo [NVCC] $@
	@$(NVCC) -o $@ $^ $(LIBS) $(DFLAGS) -I../

mcb_cycle.o: mcb_cycle.cpp $(DEPS)
	@echo [NVCC] $@
	@$(NVCC) -c -o $@ $< $(DFLAGS) -I../

Test: Test.o $(EXTRA_DEPS)
	@echo [NVCC] $@
	@$(NVCC) -o $@ $^ $(LIBS) $(DFLAGS)

mcb_cpu: mcb_non_cpu_rd.cpp $(DEPS) $(EXTRA_DEPS)
	@echo [CC] $@
	@$(CC) -o $@ $< $(CFLAGS) $(EXTRA_DEPS) $(DFLAGS) -I../

Test.o: Test.cpp $(DEPS)
	@echo [NVCC] $@
	@$(NVCC) -c -o $@ $< $(DFLAGS)

$(include)/%.o: $(include)/%.cpp $(DEPS)
	@echo [CC] $@
	@$(NVCC) $(LIBS) $(DFLAGS) -o $@ $<
.PHONY: clean

clean:
	rm -f $(GPU_DIR)/*.o $(ODIR)/*.o *~ core $(INCDIR)/*~  *.o ../include/*.o mcb_cuda Test_deg_two_remove mcb_cpu_baseline Test mcb_cpu
