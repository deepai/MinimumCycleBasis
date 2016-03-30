all: subsystem

subsystem:
	cd bicc && $(MAKE)
	cd mcb && $(MAKE)

clean:
	rm bicc/Relabeller
	rm bicc/*.o
	rm bicc/bicc_decomposition
	rm mcb/*.o
	rm mcb/mcb_cycle
	rm mcb/Test
	rm include/*.o
