all: subsystem

subsystem:
	cd bicc && $(MAKE)
	cd mcb && $(MAKE)

clean:
	rm bicc/Relabeller -f
	rm bicc/*.o -f
	rm bicc/bicc_decomposition -f 
	rm mcb/*.o -f 
	rm mcb/mcb_cycle -f
	rm mcb/Test -f 
	rm include/*.o -f
