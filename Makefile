all: subsystem

subsystem:
	cd bicc && $(MAKE)
	cd mcb && $(MAKE)

clean:
	cd bicc && $(MAKE) clean
	cd mcb && $(MAKE) clean
