IDIR =include
ODIR =obj
SRCDIR=src

EX_INCLUDE = ../include

CC=g++
CFLAGS=-I$(IDIR) -I$(EX_INCLUDE) -fopenmp -lgomp -g -w -std=c++11 -fpermissive
CXXFLAGS = -I$(IDIR) -I$(EX_INCLUDE) -fopenmp -lgomp -g -w -std=c++11 -fpermissive

DFLAGS = -DMULTI_THREAD -DPRINT -DNDEBUG

#-DMULTI_THREAD for multi-thread support
#-DPRINT to print the bccs in a file

LIBS=-lm

_DEPS = *.h 
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS)) $(patsubst %,$(EX_INCLUDE)/%,$(_DEPS))

OBJ = bicc_decomposition.o 

_EXTRA_DEPS = mmio.o Host_Timer.o
EXTRA_DEPS = $(patsubst %,$(EX_INCLUDE)/%,$(_EXTRA_DEPS))

_SRC_DEPS = mmio.cpp Host_Timer.cpp
SRC_DEPS = $(patsubst %,$(EX_INCLUDE)/%,$(_SRC_DEPS))

all: bicc_decomposition

bicc_decomposition: $(OBJ) $(EXTRA_DEPS)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS) $(DFLAGS)

bicc_decomposition.o: src/bcc_decomposition.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) $(DFLAGS)

#$(include)/%.o: $(include)/%.cpp $(DEPS)
	#$(CC) $(CFLAGS) $(LIBS) $(DFLAGS) -o $@ $< 



.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~  *.o ../include/*.o
