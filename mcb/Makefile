IDIR =include
ODIR =obj
SRCDIR=src

EX_INCLUDE = ../include

CC=g++
CFLAGS=-I$(IDIR) -I$(EX_INCLUDE) -fopenmp -lgomp -g -w -std=c++11 -fpermissive
CXXFLAGS = -I$(IDIR) -I$(EX_INCLUDE) -fopenmp -lgomp -g -w -std=c++11 -fpermissive

DFLAGS = -DMULTI_THREAD -DPRINT -DNDEBUG -DVERBOSE

#-DMULTI_THREAD for multi-thread support
#-DPRINT to print the bccs in a file

LIBS=-lm

_DEPS = *.h 
DEPS = $(patsubst %,$(EX_INCLUDE)/%,$(_DEPS))

OBJ = mcb_cycle.o 

_EXTRA_DEPS = mmio.o Host_Timer.o CsrGraphSpanningTree.o
EXTRA_DEPS = $(patsubst %,$(EX_INCLUDE)/%,$(_EXTRA_DEPS))

_SRC_DEPS = mmio.cpp Host_Timer.cpp CsrGraphSpanningTree.cpp
SRC_DEPS = $(patsubst %,$(EX_INCLUDE)/%,$(_SRC_DEPS))

all: mcb_cycle

mcb_cycle: $(OBJ) $(EXTRA_DEPS)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS) $(DFLAGS)

mcb_cycle.o: mcb_cycle.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) $(DFLAGS)

#$(include)/%.o: $(include)/%.cpp $(DEPS)
	#$(CC) $(CFLAGS) $(LIBS) $(DFLAGS) -o $@ $< 



.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~  *.o ../include/*.o
